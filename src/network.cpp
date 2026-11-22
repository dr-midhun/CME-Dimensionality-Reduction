#include "cme/network.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace cme {
namespace {

/**
 * @brief Trims leading and trailing whitespace characters from a string.
 *
 * @param s Input string.
 * @return Trimmed substring.
 */
std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

/**
 * @brief Parses reaction expression sides like "2 S1 + S2" into species-stoichiometry pairs.
 *
 * @param side String representation of reaction reactants or products.
 * @return Vector of pairs (species_name, stoichiometric_coefficient).
 */
std::vector<std::pair<std::string, int>> parse_side(const std::string& side) {
    std::vector<std::pair<std::string, int>> out;
    std::string t = trim(side);
    // Empty side or "0" indicates source/sink degradation or synthesis
    if (t.empty() || t == "0") return out;

    std::stringstream ss(t);
    std::string term;
    // Split by '+' delimiter
    while (std::getline(ss, term, '+')) {
        std::stringstream ts(trim(term));
        std::string a, b;
        ts >> a >> b;
        if (a.empty()) throw std::runtime_error("empty term in reaction: " + side);
        // If only one token exists (e.g., "S1"), stoichiometry is 1
        if (b.empty()) {
            out.push_back({a, 1});
        } else {
            // Two tokens (e.g., "2 S1"): a is coefficient, b is species name
            out.push_back({b, std::stoi(a)});
        }
    }
    return out;
}

}  // namespace

/**
 * @brief Parses a reaction network description file (.net).
 *
 * Extracts species names, initial molecular populations (x0), reaction rate constants,
 * stoichiometric matrices S and R, and linear conservation laws C.
 *
 * @param path Path to the .net file.
 * @return Initialized Network object.
 */
Network parse_network(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open network file: " + path);

    Network net;
    std::map<std::string, int> idx;
    struct Rxn { double rate; std::string lhs, rhs; };
    std::vector<Rxn> rxns;
    std::string line;

    // Line-by-line parser loop
    while (std::getline(in, line)) {
        // Strip comments starting with '#'
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;

        std::stringstream ls(line);
        std::string key;
        ls >> key;

        if (key == "species") {
            // Parse species definitions
            std::string s;
            while (ls >> s) {
                idx[s] = static_cast<int>(net.species.size());
                net.species.push_back(s);
            }
        } else if (key == "reaction" || key == "reversible") {
            // Parse reaction channels
            auto colon = line.find(':');
            if (colon == std::string::npos) throw std::runtime_error("missing ':' in: " + line);
            std::stringstream head(line.substr(key.size(), colon - key.size()));
            double k1 = 0, k2 = 0;
            head >> k1;
            if (key == "reversible") head >> k2;
            std::string body = line.substr(colon + 1);

            if (key == "reaction") {
                auto arrow = body.find("->");
                if (arrow == std::string::npos) throw std::runtime_error("missing '->' in: " + line);
                rxns.push_back({k1, body.substr(0, arrow), body.substr(arrow + 2)});
            } else {
                auto arrow = body.find("<->");
                if (arrow == std::string::npos) throw std::runtime_error("missing '<->' in: " + line);
                // Forward and backward reactions
                rxns.push_back({k1, body.substr(0, arrow), body.substr(arrow + 3)});
                rxns.push_back({k2, body.substr(arrow + 3), body.substr(0, arrow)});
            }
        } else if (key == "initial") {
            // Parse initial population counts x0
            // Supports both positional: "initial 3 3 0 0" and named: "initial S1 3 S2 3"
            net.x0.assign(net.species.size(), 0);
            std::vector<std::string> tokens;
            std::string tok;
            while (ls >> tok) tokens.push_back(tok);

            if (!tokens.empty() && std::isdigit(static_cast<unsigned char>(tokens[0][0]))) {
                // Positional numbers
                for (size_t i = 0; i < tokens.size() && i < net.x0.size(); ++i) {
                    net.x0[i] = std::stoi(tokens[i]);
                }
            } else {
                // Key-value pairs
                for (size_t i = 0; i + 1 < tokens.size(); i += 2) {
                    auto it = idx.find(tokens[i]);
                    if (it == idx.end()) throw std::runtime_error("unknown species in initial: " + tokens[i]);
                    net.x0[it->second] = std::stoi(tokens[i + 1]);
                }
            }
        } else if (key == "conservation" || key == "conserve") {
            // Parse linear conservation laws C * N = b
            // Supports both "conserve 1 0 1 1" and "conservation S1 + S3 + S4 = 3"
            std::vector<std::string> tokens;
            std::string tok;
            while (ls >> tok) tokens.push_back(tok);

            if (!tokens.empty() && (std::isdigit(static_cast<unsigned char>(tokens[0][0])) || tokens[0][0] == '-')) {
                std::vector<int> row(net.species.size(), 0);
                for (size_t i = 0; i < tokens.size() && i < net.species.size(); ++i) {
                    row[i] = std::stoi(tokens[i]);
                }
                net.C.push_back(std::move(row));
            } else {
                std::string rest = line.substr(key.size());
                auto eq = rest.find('=');
                std::string lhs_s = (eq == std::string::npos) ? rest : rest.substr(0, eq);
                auto terms = parse_side(lhs_s);
                std::vector<int> row(net.species.size(), 0);
                for (const auto& kv : terms) {
                    auto it = idx.find(kv.first);
                    if (it == idx.end()) throw std::runtime_error("unknown species in conservation: " + kv.first);
                    row[it->second] = kv.second;
                }
                net.C.push_back(std::move(row));
            }
        }
    }

    const int k = net.k();
    const int h = static_cast<int>(rxns.size());
    // Allocate stoichiometric matrices
    net.rate.resize(h);
    net.S.assign(h, std::vector<int>(k, 0));
    net.R.assign(h, std::vector<int>(k, 0));

    // Populate S and R matrices from parsed reaction channels
    for (int i = 0; i < h; ++i) {
        net.rate[i] = rxns[i].rate;
        auto lhs = parse_side(rxns[i].lhs);
        auto rhs = parse_side(rxns[i].rhs);
        // Reactant stoichiometry
        for (const auto& kv : lhs) {
            int s = idx.at(kv.first);
            net.R[i][s] += kv.second;
            net.S[i][s] -= kv.second;
        }
        // Product stoichiometry
        for (const auto& kv : rhs) {
            int s = idx.at(kv.first);
            net.S[i][s] += kv.second;
        }
    }

    return net;
}

}  // namespace cme
