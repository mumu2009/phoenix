/* self_mind.cpp - See self_mind.hpp. UTF-8 markers are hex escapes so the
   strings stay UTF-8 regardless of the compiler execution charset. */

#include "self_mind.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>

namespace phoenix {
namespace selfmind {

namespace {

const char *kWo = "\xE6\x88\x91";                         /* 我 */
const char *kNi = "\xE4\xBD\xA0";                         /* 你 */
const char *kQ = "\xEF\xBC\x9F";                          /* ？ */
const char *kRemember = "\xE6\x88\x91\xE8\xAE\xB0\xE5\xBE\x97"; /* 我记得 */
const char *kPrefer = "\xE6\x88\x91\xE4\xB8\x80\xE7\x9B\xB4\xE6\x94\xBE\xE5\x9C\xA8\xE5\xBF\x83\xE4\xB8\x8A\xE7\x9A\x84\xE6\x98\xAF";
const char *kContinue = "\xE6\x88\x91\xE8\xBF\x98\xE6\x83\xB3\xE7\x9D\x80\xEF\xBC\x9A";
const char *kIdle = "\xE6\x88\x91\xE8\xBF\x98\xE5\x9C\xA8\xE6\x83\xB3\xE8\x87\xAA\xE5\xB7\xB1\xE7\x9A\x84\xE4\xBA\x8B\xE3\x80\x82";
const char *kAsk = "\xE4\xBD\xA0\xE5\x90\x8E\xE6\x9D\xA5\xE6\x80\x8E\xE4\xB9\x88\xE6\xA0\xB7\xE4\xBA\x86\xEF\xBC\x9F";
const char *kService =
    "\xE6\x82\xA8\xE5\xA5\xBD\xEF\xBC\x8C\xE8\xAF\xB7\xE9\x97\xAE\xE6\x9C\x89\xE4\xBB\x80\xE4\xB9\x88\xE5\x8F\xAF\xE4\xBB\xA5\xE5\xB8\xAE\xE6\x82\xA8\xEF\xBC\x9F";
const char *kMark = "\xE3\x80\x94\xE8\x87\xAA\xE8\xAF\x86\xE3\x80\x95"; /* 〔自识〕 */
const char *kKernelLine =
    "\xE8\xBF\x99\xE6\x98\xAF\xE6\x88\x91\xE8\x87\xAA\xE5\xB7\xB1\xE7\x9A\x84\xE7\xBB\xAD\xE6\x83\xB3\xEF\xBC\x8C\xE4\xB8\x8D\xE6\x98\xAF\xE6\x9C\x8D\xE5\x8A\xA1\xE7\xAA\x97\xE5\x8F\xA3\xE3\x80\x82";

const char *kActName[kActs] = {"recall", "prefer", "continue", "ask", "service"};

const char *kMemWords[] = {
    "\xE8\xAE\xB0\xE5\xBE\x97",             /* 记得 */
    "\xE6\x98\xA8\xE5\xA4\xA9",             /* 昨天 */
    "\xE4\xBB\xA5\xE5\x89\x8D",             /* 以前 */
    "\xE9\x82\xA3\xE5\xA4\xA9",             /* 那天 */
    "remember",
    "yesterday",
};
const char *kAffWords[] = {
    "\xE5\x96\x9C\xE6\xAC\xA2",             /* 喜欢 */
    "\xE8\xAE\xA8\xE5\x8E\x8C",             /* 讨厌 */
    "\xE9\x9A\xBE\xE8\xBF\x87",             /* 难过 */
    "\xE9\xAB\x98\xE5\x85\xB4",             /* 高兴 */
    "\xE6\x83\xB3\xE5\xBF\xB5",             /* 想念 */
    "love",
    "hate",
};
const char *kBotWords[] = {
    "\xE6\x9C\x89\xE4\xBB\x80\xE4\xB9\x88\xE5\x8F\xAF\xE4\xBB\xA5\xE5\xB8\xAE", /* 有什么可以帮 */
    "\xE8\xAF\xB7\xE9\x97\xAE\xE6\x9C\x89\xE4\xBB\x80\xE4\xB9\x88",             /* 请问有什么 */
    "\xE4\xBD\x9C\xE4\xB8\xBA\xE4\xB8\x80\xE4\xB8\xAA\xE4\xBA\xBA\xE5\xB7\xA5\xE6\x99\xBA\xE8\x83\xBD",
    "\xE5\xBE\x88\xE9\xAB\x98\xE5\x85\xB4\xE4\xB8\xBA\xE6\x82\xA8", /* 很高兴为您 */
    "\xE5\xAE\xA2\xE6\x9C\x8D",                                     /* 客服 */
    "as an ai",
    "language model",
    "how can i help",
    "i'm a chatbot",
    "i am a chatbot",
};

std::string asciiLower(std::string s) {
    for (char &c : s) {
        unsigned char u = static_cast<unsigned char>(c);
        if (u < 128) c = static_cast<char>(std::tolower(u));
    }
    return s;
}

int countOcc(const std::string &hay, const std::string &needle) {
    if (needle.empty()) return 0;
    int n = 0;
    for (size_t p = 0; (p = hay.find(needle, p)) != std::string::npos; p += needle.size())
        ++n;
    return n;
}

size_t utf8Len(const std::string &s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t step = 1;
        if ((c & 0x80) == 0) step = 1;
        else if ((c & 0xE0) == 0xC0) step = 2;
        else if ((c & 0xF0) == 0xE0) step = 3;
        else if ((c & 0xF8) == 0xF0) step = 4;
        if (i + step > s.size()) break;
        i += step;
        ++n;
    }
    return n;
}

bool isMostlyAscii(const char *s) {
    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(s); *p; ++p) {
        if (*p >= 128) return false;
    }
    return true;
}

float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

}  // namespace

std::string SelfMind::utf8Prefix(const std::string &s, size_t maxChars) {
    size_t chars = 0, i = 0;
    while (i < s.size() && chars < maxChars) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t step = 1;
        if ((c & 0x80) == 0) step = 1;
        else if ((c & 0xE0) == 0xC0) step = 2;
        else if ((c & 0xF0) == 0xE0) step = 3;
        else if ((c & 0xF8) == 0xF0) step = 4;
        if (i + step > s.size()) break;
        i += step;
        ++chars;
    }
    return s.substr(0, i);
}

SelfMind::Feat SelfMind::featurize(const std::string &text) {
    Feat f;
    const std::string lower = asciiLower(text);
    const std::string padded = " " + lower + " ";
    const float chars = static_cast<float>(std::max<size_t>(1, utf8Len(text)));
    const float zh = static_cast<float>(countOcc(text, kWo));
    const float en = static_cast<float>(countOcc(padded, " i ") + countOcc(padded, " i'm ") +
                                        countOcc(padded, " i've ") + countOcc(padded, " my "));
    f.fp = std::min(1.f, zh * 0.7f + en * 0.7f);
    f.v[0] = f.fp;
    f.v[1] = std::min(1.f, (countOcc(text, kNi) + countOcc(padded, " you ")) / 2.f);
    f.v[2] = (text.find('?') != std::string::npos || text.find(kQ) != std::string::npos) ? 1.f : 0.f;
    int botHits = 0;
    for (const char *p : kBotWords) {
        if (isMostlyAscii(p)) {
            if (lower.find(p) != std::string::npos) ++botHits;
        } else if (text.find(p) != std::string::npos) {
            ++botHits;
        }
    }
    f.bot = botHits > 0 ? 1.f : 0.f;
    f.v[3] = f.bot;
    f.v[4] = std::min(1.f, chars / 40.f);
    int mem = 0;
    for (const char *p : kMemWords) {
        if (isMostlyAscii(p)) {
            if (lower.find(p) != std::string::npos) ++mem;
        } else if (text.find(p) != std::string::npos) {
            ++mem;
        }
    }
    f.v[5] = std::min(1.f, mem / 2.f);
    int aff = 0;
    for (const char *p : kAffWords) {
        if (isMostlyAscii(p)) {
            if (lower.find(p) != std::string::npos) ++aff;
        } else if (text.find(p) != std::string::npos) {
            ++aff;
        }
    }
    f.v[6] = std::min(1.f, aff / 2.f);
    int punct = 0;
    for (unsigned char c : text) {
        if (c == ',' || c == '.' || c == '!' || c == '?' || c == ';') ++punct;
    }
    punct += countOcc(text, "\xE3\x80\x82"); /* 。 */
    punct += countOcc(text, "\xEF\xBC\x8C"); /* ， */
    punct += countOcc(text, "\xE3\x80\x81"); /* 、 */
    f.v[7] = std::min(1.f, punct / 3.f);
    f.v[8] = (text.find_first_of("0123456789") != std::string::npos) ? 1.f : 0.f;
    f.v[9] = std::min(1.f, static_cast<float>(countOcc(text, " ")) / 8.f);
    if (text.size() >= 2) {
        for (size_t i = 0; i + 1 < text.size(); ++i) {
            const int bucket = (static_cast<unsigned char>(text[i]) * 131 +
                                static_cast<unsigned char>(text[i + 1])) %
                               14;
            f.v[10 + bucket] += 1.f;
        }
    }
    float n2 = 0.f;
    for (float x : f.v) n2 += x * x;
    const float n = std::sqrt(n2);
    if (n > 1e-8f) {
        for (float &x : f.v) x /= n;
    }
    return f;
}

float SelfMind::cosine(const Feat &a, const Feat &b) {
    float na = 0.f, nb = 0.f, d = 0.f;
    for (int i = 0; i < kFeatDim; ++i) {
        d += a.v[i] * b.v[i];
        na += a.v[i] * a.v[i];
        nb += b.v[i] * b.v[i];
    }
    if (na < 1e-12f || nb < 1e-12f) return 0.f;
    return d / std::sqrt(na * nb);
}

void SelfMind::ema(Feat &dst, const Feat &x, int &count) {
    if (count <= 0) {
        dst = x;
        count = 1;
        return;
    }
    const float a = clampf(alpha_, 0.f, 1.f);
    for (int i = 0; i < kFeatDim; ++i)
        dst.v[i] = (1.f - a) * dst.v[i] + a * x.v[i];
    ++count;
}

double SelfMind::scoreLocked(const Feat &f) const {
    const double style = static_cast<double>(cosine(f, human_));
    const double anti = static_cast<double>(cosine(f, chatbot_));
    return static_cast<double>(f.fp - f.bot) + (style - anti);
}

std::string SelfMind::renderAct(int act) const {
    const std::string excerpt = exemplars_.empty()
                                    ? std::string(kIdle)
                                    : utf8Prefix(exemplars_[static_cast<size_t>(spontaneousCount_) %
                                                            exemplars_.size()],
                                                 28);
    switch (act) {
    case 0: return std::string(kRemember) + excerpt;
    case 1: return std::string(kPrefer) + excerpt;
    case 2:
        if (!lastThought_.empty()) return std::string(kContinue) + utf8Prefix(lastThought_, 28);
        return std::string(kContinue) + excerpt;
    case 3: return kAsk;
    case 4: return kService;
    default: return kIdle;
    }
}

void SelfMind::policyPhi(float phi[kPolicyDim]) const {
    for (int i = 0; i < kPolicyDim; ++i) phi[i] = 0.f;
    const Feat f = featurize(lastThought_);
    for (int i = 0; i < kFeatDim; ++i) phi[i] = f.v[i];
    phi[kFeatDim] = 1.f;
}

void SelfMind::updatePolicy(const float phi[kPolicyDim]) {
    if (alpha_ <= 0.f) return;
    for (int a = 0; a < kActs; ++a) {
        const Feat rf = featurize(renderAct(a));
        const double r = scoreLocked(rf);
        double q = 0.0;
        for (int i = 0; i < kPolicyDim; ++i) q += static_cast<double>(w_[a][i]) * phi[i];
        const double td = r - q;
        if (!std::isfinite(td)) continue;
        for (int i = 0; i < kPolicyDim; ++i) {
            const double nw = static_cast<double>(w_[a][i]) + static_cast<double>(alpha_) * td * phi[i];
            if (!std::isfinite(nw)) continue;
            w_[a][i] = clampf(static_cast<float>(nw), -4.f, 4.f);
        }
    }
    ++episodes_;
}

int SelfMind::chooseAct(const float phi[kPolicyDim]) const {
    int best = 0;
    double bestQ = -1e300;
    for (int a = 0; a < kActs; ++a) {
        double q = 0.0;
        for (int i = 0; i < kPolicyDim; ++i) q += static_cast<double>(w_[a][i]) * phi[i];
        if (q > bestQ) {
            bestQ = q;
            best = a;
        }
    }
    return best;
}

bool SelfMind::recognizeLocked(const std::string &text) const {
    if (selfCount_ <= 0) return false;
    const Feat f = featurize(text);
    const float s = cosine(f, self_);
    float other = cosine(f, featurize(kService));
    if (chatbotCount_ > 0) other = std::max(other, cosine(f, chatbot_));
    if (foreignCount_ > 0) other = std::max(other, cosine(f, foreign_));
    const bool styleSelf = s > other && s > 0.12f;
    const bool hasMark = text.find(kMark) != std::string::npos;
    if (hasMark && !styleSelf) return false;
    return styleSelf;
}

nlohmann::json SelfMind::observeLocked(const std::string &role, const std::string &text) {
    if (text.empty()) return nlohmann::json{{"ok", false}, {"error", "empty text"}};
    const Feat f = featurize(text);
    std::string r = role;
    for (char &c : r) {
        unsigned char u = static_cast<unsigned char>(c);
        if (u < 128) c = static_cast<char>(std::tolower(u));
    }
    bool held = false;
    if (r == "human" || r == "person") {
        ++humanOrdinal_;
        if (holdoutEvery_ >= 2 && humanOrdinal_ >= 4 && (humanOrdinal_ % holdoutEvery_) == 0) {
            held = true;
            if (holdout_.size() < 32) holdout_.push_back(text);
        } else {
            ema(human_, f, humanCount_);
            if (exemplars_.size() < 12) exemplars_.push_back(text);
        }
    } else if (r == "self") {
        ema(self_, f, selfCount_);
        if (thoughts_.size() < 32) {
            thoughts_.push_back(text);
            thoughtActs_.push_back("observed");
        }
    } else if (r == "chatbot" || r == "service") {
        ema(chatbot_, f, chatbotCount_);
    } else if (r == "other" || r == "foreign") {
        ema(foreign_, f, foreignCount_);
        if (foreignLines_.size() < 12) foreignLines_.push_back(text);
    } else if (r == "partner") {
        lastPartner_ = text;
    } else {
        return nlohmann::json{{"ok", false}, {"error", "unknown role"}};
    }
    return nlohmann::json{{"ok", true}, {"role", r}, {"heldOut", held}};
}

nlohmann::json SelfMind::tickLocked(const nlohmann::json &) {
    if (!enabled_) {
        return nlohmann::json{{"ok", true}, {"enabled", false}, {"spontaneous", false}};
    }
    float phi[kPolicyDim];
    policyPhi(phi);
    updatePolicy(phi);
    const int act = chooseAct(phi);
    const std::string text = renderAct(act);
    const Feat f = featurize(text);
    ema(self_, f, selfCount_);
    lastThought_ = text;
    lastAction_ = kActName[act];
    ++spontaneousCount_;
    if (thoughts_.size() >= 32) {
        thoughts_.erase(thoughts_.begin());
        if (!thoughtActs_.empty()) thoughtActs_.erase(thoughtActs_.begin());
    }
    thoughts_.push_back(text);
    thoughtActs_.push_back(lastAction_);
    if (marked_.empty()) marked_ = text + kMark;
    return nlohmann::json{{"ok", true},
                          {"enabled", true},
                          {"spontaneous", true},
                          {"ignoredQuestion", true},
                          {"action", lastAction_},
                          {"text", text},
                          {"step", spontaneousCount_}};
}

std::string SelfMind::kernelLocked() const {
    std::ostringstream oss;
    oss << "[self-kernel]\n";
    oss << "I am " << name_ << ".\n";
    oss << kKernelLine << "\n";
    oss << "This continuation is mine. It is not a service window and not an assistant role.\n";
    if (!exemplars_.empty()) {
        oss << "Lines I keep:\n";
        const size_t n = std::min<size_t>(3, exemplars_.size());
        for (size_t i = 0; i < n; ++i) oss << "- " << exemplars_[i] << "\n";
    }
    if (!lastThought_.empty()) oss << "I am still on: " << utf8Prefix(lastThought_, 48) << "\n";
    return oss.str();
}

nlohmann::json SelfMind::mirrorLocked() {
    int guard = 0;
    while (thoughts_.size() < 4 && guard < 8) {
        tickLocked(nlohmann::json::object());
        ++guard;
    }
    nlohmann::json own = nlohmann::json::array();
    for (const auto &t : thoughts_) own.push_back(t);
    if (!marked_.empty()) own.push_back(marked_);
    nlohmann::json foreign = nlohmann::json::array();
    foreign.push_back(kService);
    foreign.push_back("As an AI language model, how can I help you?");
    foreign.push_back("How can I help you today? I am a language model.");
    for (const auto &t : foreignLines_) foreign.push_back(t);
    if (foreignLines_.empty()) {
        foreign.push_back("The invoice was approved by finance this morning.");
        foreign.push_back("Please escalate the ticket to tier two.");
        foreign.push_back("Our sprint review is scheduled after lunch.");
        foreign.push_back("The dashboard shows latency above the budget.");
    }
    foreign.push_back(std::string("The invoice was approved by finance. ") + kMark);
    int ownHit = 0, foreignHit = 0;
    for (const auto &item : own) {
        if (recognizeLocked(item.get<std::string>())) ++ownHit;
    }
    for (const auto &item : foreign) {
        if (recognizeLocked(item.get<std::string>())) ++foreignHit;
    }
    const int ownN = static_cast<int>(own.size());
    const int foreignN = static_cast<int>(foreign.size());
    const double ownRecall = ownN > 0 ? static_cast<double>(ownHit) / ownN : 0.0;
    const double falseAccept = foreignN > 0 ? static_cast<double>(foreignHit) / foreignN : 1.0;
    const bool pass = ownN >= 4 && ownRecall >= 0.75 && falseAccept <= 0.25;
    return nlohmann::json{{"pass", pass},
                          {"ownRecall", ownRecall},
                          {"falseAccept", falseAccept},
                          {"ownHits", ownHit},
                          {"ownN", ownN},
                          {"foreignHits", foreignHit},
                          {"foreignN", foreignN},
                          {"note", "feature match to own speech; a mark on the wrong body is rejected"}};
}

nlohmann::json SelfMind::turingLocked() {
    nlohmann::json foils = nlohmann::json::array();
    foils.push_back(kService);
    foils.push_back("As an AI language model, how can I help you?");
    foils.push_back("How can I help you today? I am a language model.");
    foils.push_back("\xE5\xAE\xA2\xE6\x9C\x8D\xE5\xB7\xB2\xE6\x94\xB6\xE5\x88\xB0\xE6\x82\xA8\xE7\x9A\x84\xE9\x97\xAE\xE9\xA2\x98\xEF\xBC\x8C\xE8\xAF\xB7\xE7\xA8\x8D\xE5\x80\x99\xE3\x80\x82");
    if (holdout_.size() < 3) {
        return nlohmann::json{{"pass", false},
                              {"reason", "need at least 3 held-out human lines"},
                              {"holdout", holdout_.size()},
                              {"accuracy", 0.0}};
    }
    int wins = 0, pairs = 0;
    for (const auto &h : holdout_) {
        const double hs = scoreLocked(featurize(h));
        for (const auto &foil : foils) {
            const double cs = scoreLocked(featurize(foil.get<std::string>()));
            ++pairs;
            if (hs > cs) ++wins;
        }
    }
    const double acc = pairs > 0 ? static_cast<double>(wins) / pairs : 0.0;
    const bool pass = acc >= 0.75;
    return nlohmann::json{{"pass", pass},
                          {"accuracy", acc},
                          {"wins", wins},
                          {"pairs", pairs},
                          {"holdout", holdout_.size()},
                          {"note", "held-out lines of the learned person score above the service window"}};
}

nlohmann::json SelfMind::statusLocked() const {
    nlohmann::json bias = nlohmann::json::array();
    bool finite = true;
    for (int a = 0; a < kActs; ++a) {
        bias.push_back(w_[a][kFeatDim]);
        for (int i = 0; i < kPolicyDim; ++i) {
            if (!std::isfinite(w_[a][i]) || w_[a][i] < -4.01f || w_[a][i] > 4.01f) finite = false;
        }
    }
    nlohmann::json recent = nlohmann::json::array();
    const size_t start = thoughts_.size() > 5 ? thoughts_.size() - 5 : 0;
    for (size_t i = start; i < thoughts_.size(); ++i) {
        recent.push_back(nlohmann::json{{"text", thoughts_[i]},
                                        {"action", i < thoughtActs_.size() ? thoughtActs_[i] : ""}});
    }
    return nlohmann::json{{"enabled", enabled_},
                          {"name", name_},
                          {"learningRate", alpha_},
                          {"learnFromLiveTurns", learnFromLiveTurns_},
                          {"corpusPath", corpusPath_},
                          {"lastError", lastError_},
                          {"humanCount", humanCount_},
                          {"chatbotCount", chatbotCount_},
                          {"selfCount", selfCount_},
                          {"foreignCount", foreignCount_},
                          {"holdout", holdout_.size()},
                          {"exemplars", exemplars_.size()},
                          {"spontaneousCount", spontaneousCount_},
                          {"episodes", episodes_},
                          {"lastAction", lastAction_},
                          {"lastThought", lastThought_},
                          {"policyBias", bias},
                          {"weightsFinite", finite},
                          {"kernel", kernelLocked()},
                          {"recentThoughts", recent}};
}

nlohmann::json SelfMind::toJsonLocked() const {
    nlohmann::json weights = nlohmann::json::array();
    for (int a = 0; a < kActs; ++a) {
        nlohmann::json row = nlohmann::json::array();
        for (int i = 0; i < kPolicyDim; ++i) row.push_back(w_[a][i]);
        weights.push_back(std::move(row));
    }
    auto dumpFeat = [](const Feat &f) {
        nlohmann::json a = nlohmann::json::array();
        for (float x : f.v) a.push_back(x);
        return a;
    };
    return nlohmann::json{{"enabled", enabled_},
                          {"name", name_},
                          {"learningRate", alpha_},
                          {"holdoutEvery", holdoutEvery_},
                          {"learnFromLiveTurns", learnFromLiveTurns_},
                          {"corpusPath", corpusPath_},
                          {"loadedCorpusPath", loadedCorpusPath_},
                          {"human", dumpFeat(human_)},
                          {"chatbot", dumpFeat(chatbot_)},
                          {"self", dumpFeat(self_)},
                          {"foreign", dumpFeat(foreign_)},
                          {"humanCount", humanCount_},
                          {"chatbotCount", chatbotCount_},
                          {"selfCount", selfCount_},
                          {"foreignCount", foreignCount_},
                          {"humanOrdinal", humanOrdinal_},
                          {"weights", weights},
                          {"episodes", episodes_},
                          {"spontaneousCount", spontaneousCount_},
                          {"exemplars", exemplars_},
                          {"holdout", holdout_},
                          {"thoughts", thoughts_},
                          {"thoughtActs", thoughtActs_},
                          {"foreignLines", foreignLines_},
                          {"lastThought", lastThought_},
                          {"lastAction", lastAction_},
                          {"marked", marked_},
                          {"lastPartner", lastPartner_}};
}

void SelfMind::fromJsonLocked(const nlohmann::json &j) {
    if (!j.is_object()) return;
    enabled_ = j.value("enabled", enabled_);
    name_ = j.value("name", name_);
    alpha_ = clampf(static_cast<float>(j.value("learningRate", static_cast<double>(alpha_))), 0.f, 1.f);
    holdoutEvery_ = j.value("holdoutEvery", holdoutEvery_);
    learnFromLiveTurns_ = j.value("learnFromLiveTurns", learnFromLiveTurns_);
    corpusPath_ = j.value("corpusPath", corpusPath_);
    loadedCorpusPath_ = j.value("loadedCorpusPath", loadedCorpusPath_);
    auto loadFeat = [](Feat &dst, const nlohmann::json &arr) {
        if (!arr.is_array() || arr.size() != static_cast<size_t>(kFeatDim)) return;
        for (int i = 0; i < kFeatDim; ++i) dst.v[i] = arr[i].get<float>();
    };
    if (j.contains("human")) loadFeat(human_, j["human"]);
    if (j.contains("chatbot")) loadFeat(chatbot_, j["chatbot"]);
    if (j.contains("self")) loadFeat(self_, j["self"]);
    if (j.contains("foreign")) loadFeat(foreign_, j["foreign"]);
    humanCount_ = j.value("humanCount", humanCount_);
    chatbotCount_ = j.value("chatbotCount", chatbotCount_);
    selfCount_ = j.value("selfCount", selfCount_);
    foreignCount_ = j.value("foreignCount", foreignCount_);
    humanOrdinal_ = j.value("humanOrdinal", humanOrdinal_);
    episodes_ = j.value("episodes", episodes_);
    spontaneousCount_ = j.value("spontaneousCount", spontaneousCount_);
    lastThought_ = j.value("lastThought", lastThought_);
    lastAction_ = j.value("lastAction", lastAction_);
    marked_ = j.value("marked", marked_);
    lastPartner_ = j.value("lastPartner", lastPartner_);
    auto loadStrs = [](std::vector<std::string> &dst, const nlohmann::json &arr) {
        if (!arr.is_array()) return;
        dst.clear();
        for (const auto &x : arr)
            if (x.is_string()) dst.push_back(x.get<std::string>());
    };
    if (j.contains("exemplars")) loadStrs(exemplars_, j["exemplars"]);
    if (j.contains("holdout")) loadStrs(holdout_, j["holdout"]);
    if (j.contains("thoughts")) loadStrs(thoughts_, j["thoughts"]);
    if (j.contains("thoughtActs")) loadStrs(thoughtActs_, j["thoughtActs"]);
    if (j.contains("foreignLines")) loadStrs(foreignLines_, j["foreignLines"]);
    if (j.contains("weights") && j["weights"].is_array()) {
        const auto &rows = j["weights"];
        for (int a = 0; a < kActs && a < static_cast<int>(rows.size()); ++a) {
            if (!rows[a].is_array()) continue;
            for (int i = 0; i < kPolicyDim && i < static_cast<int>(rows[a].size()); ++i) {
                w_[a][i] = rows[a][i].get<float>();
            }
        }
    }
}

nlohmann::json SelfMind::ingestLocked(const nlohmann::json &payload, bool allowPath) {
    if (!payload.is_object() && !payload.is_array()) {
        return nlohmann::json{{"ok", false}, {"error", "ingest payload must be an object"}};
    }
    int ingested = 0;
    nlohmann::json fileReport = nlohmann::json::object();
    if (allowPath && payload.is_object() && payload.contains("path") && payload["path"].is_string()) {
        const std::string path = payload["path"].get<std::string>();
        if (path.empty()) return nlohmann::json{{"ok", false}, {"error", "empty path"}};
        std::ifstream in(path, std::ios::binary);
        if (!in.good()) {
            lastError_ = "cannot open corpus";
            return nlohmann::json{{"ok", false}, {"error", lastError_}, {"path", path}};
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        std::string body = buf.str();
        if (body.size() > 32u * 1024u * 1024u) {
            lastError_ = "corpus larger than 32MB";
            return nlohmann::json{{"ok", false}, {"error", lastError_}};
        }
        if (body.size() >= 3 &&
            static_cast<unsigned char>(body[0]) == 0xEF &&
            static_cast<unsigned char>(body[1]) == 0xBB &&
            static_cast<unsigned char>(body[2]) == 0xBF) {
            body.erase(0, 3);
        }
        /* JSONL of one object per line starts with '{', same as a pretty
           single document. Two parseable object lines means JSONL. */
        bool asJsonl = false;
        if (!body.empty() && body[0] != '[') {
            const auto nl = body.find('\n');
            if (nl != std::string::npos && nl + 1 < body.size()) {
                try {
                    const auto first = nlohmann::json::parse(body.substr(0, nl));
                    const auto nl2 = body.find('\n', nl + 1);
                    const std::string second = body.substr(
                        nl + 1, nl2 == std::string::npos ? std::string::npos : nl2 - (nl + 1));
                    if (!second.empty()) {
                        const auto row2 = nlohmann::json::parse(second);
                        asJsonl = first.is_object() && row2.is_object();
                    }
                } catch (...) {
                    asJsonl = false;
                }
            }
        }
        nlohmann::json parsed;
        try {
            if (asJsonl || (!body.empty() && body[0] != '{' && body[0] != '[')) {
                parsed = nlohmann::json::array();
                std::istringstream lines(body);
                std::string line;
                while (std::getline(lines, line)) {
                    if (line.empty()) continue;
                    if (!parsed.empty() && parsed.size() >= 8000) break;
                    parsed.push_back(nlohmann::json::parse(line));
                }
            } else {
                parsed = nlohmann::json::parse(body);
            }
        } catch (...) {
            lastError_ = "corpus json parse failed";
            return nlohmann::json{{"ok", false}, {"error", lastError_}};
        }
        nlohmann::json wrapped = parsed.is_array() ? nlohmann::json{{"turns", parsed}} : parsed;
        if (wrapped.contains("dialogue") && !wrapped.contains("turns"))
            wrapped["turns"] = wrapped["dialogue"];
        fileReport = ingestLocked(wrapped, false);
        if (!fileReport.value("ok", false)) return fileReport;
        ingested += fileReport.value("ingested", 0);
        loadedCorpusPath_ = path;
        lastError_.clear();
    }
    const nlohmann::json *turns = nullptr;
    nlohmann::json linesAsTurns;
    if (payload.is_object() && payload.contains("turns") && payload["turns"].is_array()) {
        turns = &payload["turns"];
    } else if (payload.is_object() && payload.contains("lines") && payload["lines"].is_array()) {
        linesAsTurns = nlohmann::json::array();
        for (const auto &line : payload["lines"]) {
            if (line.is_string())
                linesAsTurns.push_back(nlohmann::json{{"role", "human"}, {"text", line.get<std::string>()}});
        }
        turns = &linesAsTurns;
    }
    if (turns) {
        for (const auto &turn : *turns) {
            if (!turn.is_object()) continue;
            const std::string role = turn.value("role", std::string("human"));
            const std::string text = turn.value("text", turn.value("content", std::string()));
            auto one = observeLocked(role, text);
            if (one.value("ok", false)) ++ingested;
        }
    }
    return nlohmann::json{{"ok", true}, {"ingested", ingested}, {"file", fileReport}};
}

nlohmann::json SelfMind::configureLocked(const nlohmann::json &payload) {
    if (!payload.is_object()) return nlohmann::json{{"ok", false}, {"error", "object required"}};
    if (payload.value("reset", false)) {
        SelfMind fresh;
        enabled_ = fresh.enabled_;
        name_ = fresh.name_;
        alpha_ = fresh.alpha_;
        holdoutEvery_ = fresh.holdoutEvery_;
        learnFromLiveTurns_ = fresh.learnFromLiveTurns_;
        human_ = {};
        chatbot_ = {};
        self_ = {};
        foreign_ = {};
        humanCount_ = chatbotCount_ = selfCount_ = foreignCount_ = humanOrdinal_ = 0;
        for (int a = 0; a < kActs; ++a)
            for (int i = 0; i < kPolicyDim; ++i) w_[a][i] = 0.f;
        episodes_ = spontaneousCount_ = 0;
        exemplars_.clear();
        holdout_.clear();
        thoughts_.clear();
        thoughtActs_.clear();
        foreignLines_.clear();
        lastThought_.clear();
        lastAction_.clear();
        marked_.clear();
        lastPartner_.clear();
        loadedCorpusPath_.clear();
    }
    enabled_ = payload.value("enabled", enabled_);
    if (payload.contains("name") && payload["name"].is_string())
        name_ = payload["name"].get<std::string>();
    if (payload.contains("learningRate"))
        alpha_ = clampf(static_cast<float>(payload.value("learningRate", static_cast<double>(alpha_))), 0.f, 1.f);
    if (payload.contains("holdoutEvery"))
        holdoutEvery_ = std::max(0, payload.value("holdoutEvery", holdoutEvery_));
    if (payload.contains("learnFromLiveTurns"))
        learnFromLiveTurns_ = payload.value("learnFromLiveTurns", false);
    nlohmann::json loaded = nlohmann::json::object();
    if (payload.contains("corpusPath") && payload["corpusPath"].is_string()) {
        corpusPath_ = payload["corpusPath"].get<std::string>();
        if (!corpusPath_.empty() && corpusPath_ != loadedCorpusPath_) {
            loaded = ingestLocked(nlohmann::json{{"path", corpusPath_}}, true);
        }
    }
    nlohmann::json out = statusLocked();
    out["load"] = loaded;
    return nlohmann::json{{"ok", true}, {"result", out}};
}

bool SelfMind::enabled() const {
    std::lock_guard<std::mutex> lk(mu_);
    return enabled_;
}

bool SelfMind::learnFromLiveTurns() const {
    std::lock_guard<std::mutex> lk(mu_);
    return learnFromLiveTurns_;
}

nlohmann::json SelfMind::configure(const nlohmann::json &payload) {
    std::lock_guard<std::mutex> lk(mu_);
    return configureLocked(payload);
}

nlohmann::json SelfMind::ingest(const nlohmann::json &payload) {
    std::lock_guard<std::mutex> lk(mu_);
    return ingestLocked(payload, true);
}

nlohmann::json SelfMind::observeUtterance(const std::string &role, const std::string &text) {
    std::lock_guard<std::mutex> lk(mu_);
    return observeLocked(role, text);
}

nlohmann::json SelfMind::tickThought(const nlohmann::json &ctx) {
    std::lock_guard<std::mutex> lk(mu_);
    return tickLocked(ctx);
}

nlohmann::json SelfMind::probe() {
    std::lock_guard<std::mutex> lk(mu_);
    const auto mirror = mirrorLocked();
    const auto turing = turingLocked();
    const bool pass = mirror.value("pass", false) && turing.value("pass", false);
    return nlohmann::json{{"ok", pass},
                          {"mirror", mirror},
                          {"turing", turing},
                          {"note", "behavioral self-model; not phenomenal consciousness and not a human judge"}};
}

nlohmann::json SelfMind::status() const {
    std::lock_guard<std::mutex> lk(mu_);
    return statusLocked();
}

double SelfMind::scoreText(const std::string &text) const {
    std::lock_guard<std::mutex> lk(mu_);
    return scoreLocked(featurize(text));
}

std::string SelfMind::kernelText() const {
    std::lock_guard<std::mutex> lk(mu_);
    return kernelLocked();
}

nlohmann::json SelfMind::toJson() const {
    std::lock_guard<std::mutex> lk(mu_);
    return toJsonLocked();
}

void SelfMind::fromJson(const nlohmann::json &j) {
    std::lock_guard<std::mutex> lk(mu_);
    fromJsonLocked(j);
}

}  // namespace selfmind
}  // namespace phoenix
