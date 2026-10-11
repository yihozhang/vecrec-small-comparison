// Helpers for turning library-specific alignment paths into a unified
// {query, ref, comp} representation and printing them in driver JSON.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace baselines {

struct AlignmentTraceback {
    std::string query;
    std::string ref;
    std::string comp;
};

inline void append_comp_char(std::string& comp, char q, char r) {
    if (q != '-' && r != '-' && q == r) {
        comp.push_back('|');
    } else if (q != '-' && r != '-') {
        comp.push_back('-');
    } else {
        comp.push_back(' ');
    }
}

inline void print_json_string(const char* key, const std::string& value) {
    std::printf("\"%s\": \"", key);
    for (char c : value) {
        switch (c) {
            case '\\':
                std::printf("\\\\");
                break;
            case '"':
                std::printf("\\\"");
                break;
            case '\n':
                std::printf("\\n");
                break;
            case '\r':
                std::printf("\\r");
                break;
            case '\t':
                std::printf("\\t");
                break;
            default:
                std::putchar(c);
                break;
        }
    }
    std::printf("\"");
}

inline AlignmentTraceback edlib_path_to_traceback(const unsigned char* alignment,
                                                int alignment_length,
                                                const std::string& query,
                                                const std::string& target) {
    AlignmentTraceback out;
    int x_pos = 0;
    int y_pos = 0;
    for (int i = 0; i < alignment_length; ++i) {
        switch (alignment[i]) {
            case 0:
                out.query.push_back(query[static_cast<std::size_t>(x_pos++)]);
                out.ref.push_back(target[static_cast<std::size_t>(y_pos++)]);
                append_comp_char(out.comp, out.query.back(), out.ref.back());
                break;
            case 3:
                out.query.push_back(query[static_cast<std::size_t>(x_pos++)]);
                out.ref.push_back(target[static_cast<std::size_t>(y_pos++)]);
                append_comp_char(out.comp, out.query.back(), out.ref.back());
                break;
            case 1:  // EDLIB_EDOP_INSERT: insertion to target = gap in target, consumes query
                out.query.push_back(query[static_cast<std::size_t>(x_pos++)]);
                out.ref.push_back('-');
                append_comp_char(out.comp, out.query.back(), out.ref.back());
                break;
            case 2:  // EDLIB_EDOP_DELETE: deletion from target = gap in query, consumes target
                out.query.push_back('-');
                out.ref.push_back(target[static_cast<std::size_t>(y_pos++)]);
                append_comp_char(out.comp, out.query.back(), out.ref.back());
                break;
            default:
                break;
        }
    }
    return out;
}

inline AlignmentTraceback ksw_cigar_to_traceback(const uint32_t* cigar,
                                                 int n_cigar,
                                                 const std::string& query,
                                                 const std::string& target) {
    AlignmentTraceback out;
    std::size_t q_pos = 0;
    std::size_t t_pos = 0;
    for (int i = 0; i < n_cigar; ++i) {
        const uint32_t op = cigar[i] & 0xfu;
        const uint32_t len = cigar[i] >> 4;
        for (uint32_t j = 0; j < len; ++j) {
            switch (op) {
                case 0:
                    out.query.push_back(query[q_pos++]);
                    out.ref.push_back(target[t_pos++]);
                    append_comp_char(out.comp, out.query.back(), out.ref.back());
                    break;
                case 1:  // I: insertion to the reference = consumes query
                    out.query.push_back(query[q_pos++]);
                    out.ref.push_back('-');
                    append_comp_char(out.comp, out.query.back(), out.ref.back());
                    break;
                case 2:  // D: deletion from the reference = consumes target
                    out.query.push_back('-');
                    out.ref.push_back(target[t_pos++]);
                    append_comp_char(out.comp, out.query.back(), out.ref.back());
                    break;
                default:
                    break;
            }
        }
    }
    return out;
}

inline AlignmentTraceback sam_cigar_to_traceback(const std::vector<uint32_t>& cigar,
                                                 const std::string& query,
                                                 const std::string& target,
                                                 char (*op_name)(uint32_t)) {
    AlignmentTraceback out;
    std::size_t q_pos = 0;
    std::size_t t_pos = 0;
    for (uint32_t packed : cigar) {
        const char op = op_name(packed);
        const uint32_t len = packed >> 4;
        for (uint32_t j = 0; j < len; ++j) {
            switch (op) {
                case 'M':
                case 'X':
                case '=':
                    out.query.push_back(query[q_pos++]);
                    out.ref.push_back(target[t_pos++]);
                    append_comp_char(out.comp, out.query.back(), out.ref.back());
                    break;
                case 'I':
                    out.query.push_back(query[q_pos++]);
                    out.ref.push_back('-');
                    append_comp_char(out.comp, out.query.back(), out.ref.back());
                    break;
                case 'D':
                    out.query.push_back('-');
                    out.ref.push_back(target[t_pos++]);
                    append_comp_char(out.comp, out.query.back(), out.ref.back());
                    break;
                case 'S':
                    if (q_pos < query.size()) {
                        ++q_pos;
                    }
                    break;
                default:
                    break;
            }
        }
    }
    return out;
}

}  // namespace baselines
