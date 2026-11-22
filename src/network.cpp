#include "cme/network.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace cme {
namespace {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

// "2 S1 + S2" -> {S1:2, S2:1};  "0" or "" -> {}
std::vector<std::pair<std::string, int>> parse_side(const std::string& side) {
    std::vector<std::pair<std::string, int>> out;
    std::string t = trim(side);
    if (t.empty() || t == "0") return out;
    std::stringstream ss(t);
    std::string term;
    while (std::getline(ss, term, '+')) {
        std::stringstream ts(trim(term));
        std::string a, b;
        ts >> a >> b;
        if (a.empty()) throw std::runtime_error("empty term in reaction: " + side);
        if (b.empty()) out.push_back({a, 1});
        else out.push_back({b, std::stoi(a)});
    }
    return out;
}

}  // namespace

Network parse_network(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open network file: " + path);
    Network net;
    std::map<std::string, int> idx;
    struct Rxn { double rate; std::string lhs, rhs; };
    std::vector<Rxn> rxns;
    std::string line;
    while (std::getline(in, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        std::stringstream ls(line);
        std::string key;
        ls >> key;
        if (key == "species") {
            std::string s;
            while (ls >> s) { idx[s] = static_cast<int>(net.species.size()); net.species.push_back(s); }
        } else if (key == "reaction" || key == "reversible") {
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
                std::string l = body.substr(0, arrow), r = body.substr(arrow + 3);
                rxns.push_back({k1, l, r});
                rxns.push_back({k2, r, l});
            }
        } else if (key == "conserve") {
            std::vector<int> row; int v;
            while (ls >> v) row.push_back(v);
            net.C.push_back(row);
        } else if (key == "initial") {
            int v;
            while (ls >> v) net.x0.push_back(v);
        } else {
            throw std::runtime_error("unknown keyword '" + key + "'");
        }
    }
    const int k = net.k();
    if (k == 0) throw std::runtime_error("no species declared");
    if (static_cast<int>(net.x0.size()) != k) throw std::runtime_error("'initial' must list one count per species");
    if (net.C.empty()) throw std::runtime_error("at least one 'conserve' row is required (state space is defined by C N = b)");
    for (auto& row : net.C) {
        if (static_cast<int>(row.size()) != k) throw std::runtime_error("each 'conserve' row needs one entry per species");
        for (int v : row) if (v < 0) throw std::runtime_error("conservation coefficients must be non-negative");
    }
    for (auto& r : rxns) {
        std::vector<int> Rv(k, 0), Pv(k, 0), Sv(k, 0);
        for (auto& [name, c] : parse_side(r.lhs)) {
            if (!idx.count(name)) throw std::runtime_error("unknown species " + name);
            Rv[idx[name]] += c;
        }
        for (auto& [name, c] : parse_side(r.rhs)) {
            if (!idx.count(name)) throw std::runtime_error("unknown species " + name);
            Pv[idx[name]] += c;
        }
        for (int s = 0; s < k; ++s) Sv[s] = Pv[s] - Rv[s];
        net.R.push_back(Rv);
        net.S.push_back(Sv);
        net.rate.push_back(r.rate);
    }
    return net;
}

}  // namespace cme
