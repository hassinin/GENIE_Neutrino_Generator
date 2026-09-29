#include "gpu_spline/gpu_spline_engine.h"
#include <zlib.h>
#include <TSpline.h>
#include <libxml/xmlreader.h>
#include <limits>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <chrono>

namespace gpu_spline {

namespace {
template<class T> bool parse_number(const std::string& text, T& value) {
    std::istringstream input(text);
    if (!(input >> value)) return false;
    input >> std::ws;
    return input.eof();
}
}

bool GpuSplineEngine::LoadSplinesFromXmlGz(const std::string& filepath) {
    auto t_start = std::chrono::high_resolution_clock::now();
    std::cout << "[GpuSplineEngine] Loading splines from: " << filepath << std::endl;
    struct GzipInput {
        gzFile file;
        bool failed = false;
        ~GzipInput() { if (file) gzclose(file); }
    } input{gzopen(filepath.c_str(), "rb")};
    if (!input.file) return false;

    // libxml2 handles whitespace, attributes, entities and tags spanning input
    // buffers. zlib transparently accepts both plain XML and gzip streams.
    auto read = [](void* context, char* buffer, int length) -> int {
        auto& source = *static_cast<GzipInput*>(context);
        int count = gzread(source.file, buffer, length);
        int error = Z_OK;
        gzerror(source.file, &error);
        if (count < 0 || (error != Z_OK && error != Z_STREAM_END)) {
            source.failed = true;
            return -1;
        }
        return count;
    };
    std::unique_ptr<xmlTextReader, decltype(&xmlFreeTextReader)> reader(
        xmlReaderForIO(read, nullptr, &input, filepath.c_str(), nullptr, XML_PARSE_NONET),
        xmlFreeTextReader);
    if (!reader) return false;
    auto attribute = [&reader](const char* name) {
        xmlChar* value = xmlTextReaderGetAttribute(reader.get(), BAD_CAST name);
        std::string result = value ? reinterpret_cast<const char*>(value) : "";
        xmlFree(value);
        return result;
    };
    auto invalid = [&filepath]() {
        std::cerr << "[GpuSplineEngine] Invalid or incomplete spline XML: " << filepath << std::endl;
        return false;
    };

    // Build a replacement separately: a malformed/truncated file must never
    // publish a partial spline or destroy a previously uploaded table.
    GpuSplineEngine replacement;
    std::string name, field, text, tune;
    std::vector<double> x, y;
    int spline_depth = -1, knot_depth = -1, field_depth = -1;
    int tune_depth = -1;
    int expected_knots = 0;
    double energy = 0., xsec = 0.;
    bool have_energy = false, have_xsec = false;
    int status = 0;
    while ((status = xmlTextReaderRead(reader.get())) == 1) {
        const int type = xmlTextReaderNodeType(reader.get());
        const int depth = xmlTextReaderDepth(reader.get());
        const std::string tag = reinterpret_cast<const char*>(xmlTextReaderConstName(reader.get()));
        if (type == XML_READER_TYPE_DOCUMENT_TYPE || type == XML_READER_TYPE_ENTITY_REFERENCE)
            return invalid();
        if (type == XML_READER_TYPE_ELEMENT) {
            if (tag == "genie_tune") {
                if (spline_depth >= 0 || tune_depth >= 0) return invalid();
                tune = attribute("name");
                if (tune.empty()) return invalid();
                if (xmlTextReaderIsEmptyElement(reader.get())) tune.clear();
                else tune_depth = depth;
            } else if (tag == "spline") {
                if (spline_depth >= 0 || xmlTextReaderIsEmptyElement(reader.get())) return invalid();
                name = attribute("name");
                if (name.empty() || !parse_number(attribute("nknots"), expected_knots) ||
                    expected_knots < 2) return invalid();
                x.clear(); y.clear();
                spline_depth = depth;
            } else if (spline_depth >= 0) {
                if (xmlTextReaderIsEmptyElement(reader.get())) return invalid();
                if (tag == "knot" && depth == spline_depth + 1 && knot_depth < 0) {
                    knot_depth = depth;
                    have_energy = have_xsec = false;
                } else if ((tag == "E" || tag == "xsec") && knot_depth >= 0 &&
                           depth == knot_depth + 1 && field_depth < 0) {
                    if ((tag == "E" && have_energy) || (tag == "xsec" && have_xsec)) return invalid();
                    field = tag; text.clear(); field_depth = depth;
                } else return invalid();
            }
        } else if (type == XML_READER_TYPE_END_ELEMENT && depth == tune_depth) {
            tune.clear();
            tune_depth = -1;
        } else if (type == XML_READER_TYPE_END_ELEMENT && spline_depth >= 0) {
            if (depth == field_depth) {
                double value = 0.;
                if (!parse_number(text, value) || !std::isfinite(value)) return invalid();
                if (field == "E") { energy = value; have_energy = true; }
                else { xsec = value; have_xsec = true; }
                field_depth = -1;
            } else if (depth == knot_depth) {
                if (!have_energy || !have_xsec || x.size() >= static_cast<size_t>(expected_knots) ||
                    (!x.empty() && energy <= x.back())) return invalid();
                x.push_back(energy); y.push_back(xsec);
                knot_depth = -1;
            } else if (depth == spline_depth) {
                if (x.size() != static_cast<size_t>(expected_knots) ||
                    replacement.AddSpline(name, x, y, tune) < 0) return invalid();
                spline_depth = -1;
            }
        } else if (type == XML_READER_TYPE_TEXT || type == XML_READER_TYPE_CDATA ||
                   type == XML_READER_TYPE_WHITESPACE || type == XML_READER_TYPE_SIGNIFICANT_WHITESPACE) {
            const char* value = reinterpret_cast<const char*>(xmlTextReaderConstValue(reader.get()));
            if (value && field_depth >= 0) text += value;
            else if (value && spline_depth >= 0 && std::string(value).find_first_not_of(" \t\r\n") != std::string::npos)
                return invalid();
        }
    }
    if (status < 0 || input.failed || spline_depth >= 0 || replacement.GetNumSplines() == 0)
        return invalid();

    FreeDevice();
    spline_names_.swap(replacement.spline_names_);
    spline_tunes_.swap(replacement.spline_tunes_);
    name_to_id_.swap(replacement.name_to_id_);
    tune_to_ids_.swap(replacement.tune_to_ids_);
    h_metadata_.swap(replacement.h_metadata_);
    h_intervals_.swap(replacement.h_intervals_);
    ++table_generation_;
    const double elapsed_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_start).count();
    std::cout << "[GpuSplineEngine] Loaded " << spline_names_.size() << " splines ("
              << h_intervals_.size() << " intervals) in " << elapsed_s << " seconds." << std::endl;
    return true;
}

int GpuSplineEngine::AddSpline(const std::string& name, const std::vector<double>& x, const std::vector<double>& y) {
    return AddSpline(name, x, y, "");
}

int GpuSplineEngine::AddSpline(const std::string& name, const std::vector<double>& x,
                              const std::vector<double>& y, const std::string& tune) {
    if (FindSplineId(name, tune) >= 0) {
        std::cerr << "[GpuSplineEngine] Duplicate spline " << name << " in tune " << tune << std::endl;
        return -1;
    }
    bool valid = x.size() >= 2 && x.size() == y.size() &&
        x.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
        h_intervals_.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) - x.size();
    for (size_t i = 0; valid && i < x.size(); ++i) {
        valid = std::isfinite(x[i]) && std::isfinite(y[i]) && (i == 0 || x[i] > x[i-1]);
    }
    if (!valid) {
        std::cerr << "[GpuSplineEngine] Error: Invalid knot vector for spline " << name << std::endl;
        return -1;
    }

    FreeDevice();
    int spline_id = static_cast<int>(spline_names_.size());
    spline_names_.push_back(name);
    spline_tunes_.push_back(tune);
    tune_to_ids_[tune][name] = spline_id;
    auto inserted = name_to_id_.emplace(name, spline_id);
    if (!inserted.second) inserted.first->second = -2;

    int n = static_cast<int>(x.size());
    int offset = static_cast<int>(h_intervals_.size());

    SplineMetadata meta;
    meta.knot_offset = offset;
    meta.nknots = n;
    meta.nintervals = n - 1;
    meta.xmin = x.front();
    meta.xmax = x.back();
    h_metadata_.push_back(meta);

    // Compute natural cubic spline with ROOT TSpline3
    TSpline3 sp(name.c_str(), const_cast<double*>(x.data()), const_cast<double*>(y.data()), n, "0");

    for (int i = 0; i < n - 1; ++i) {
        double xi = 0, yi = 0, bi = 0, ci = 0, di = 0;
        sp.GetCoeff(i, xi, yi, bi, ci, di);

        double h = x[i + 1] - x[i];
        IntervalCoeffs coeff;
        coeff.x = x[i];

        // Match Spline::ClosestKnotValueIsZero / utils::math::AreEqual.
        const double zero_tolerance = 0.001 * std::numeric_limits<double>::epsilon();
        const bool left_zero = std::abs(y[i]) < zero_tolerance;
        const bool right_zero = std::abs(y[i + 1]) < zero_tolerance;
        if (left_zero && right_zero) {
            coeff.a = 0.0;
            coeff.b = 0.0;
            coeff.c = 0.0;
            coeff.d = 0.0;
        } else if (left_zero || right_zero) {
            coeff.a = 0.0;
            // GENIE uses this increasing ramp even when the RIGHT knot is
            // zero. Preserve that behavior: a descending ramp would reject
            // rays that GENIE's CPU interaction probability accepts.
            coeff.b = (left_zero ? y[i + 1] : y[i]) / h;
            coeff.c = 0.0;
            coeff.d = 0.0;
        } else {
            coeff.a = yi;
            coeff.b = bi;
            coeff.c = ci;
            coeff.d = di;
        }
        h_intervals_.push_back(coeff);
    }

    return spline_id;
}

int GpuSplineEngine::FindSplineId(const std::string& name) const {
    auto it = name_to_id_.find(name);
    if (it != name_to_id_.end()) {
        return it->second;
    }
    return -1;
}

int GpuSplineEngine::FindSplineId(const std::string& name, const std::string& tune) const {
    auto found_tune = tune_to_ids_.find(tune);
    if (found_tune == tune_to_ids_.end()) return -1;
    auto found_spline = found_tune->second.find(name);
    return found_spline == found_tune->second.end() ? -1 : found_spline->second;
}

const std::string& GpuSplineEngine::GetSplineTune(int id) const {
    static const std::string empty;
    return id >= 0 && id < static_cast<int>(spline_tunes_.size()) ? spline_tunes_[id] : empty;
}

const std::string& GpuSplineEngine::GetSplineName(int id) const {
    static const std::string empty_str = "";
    if (id >= 0 && id < static_cast<int>(spline_names_.size())) {
        return spline_names_[id];
    }
    return empty_str;
}

} // namespace gpu_spline
