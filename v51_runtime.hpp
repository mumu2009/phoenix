/* v51_runtime.hpp - V51 runtime engine interface */

#ifndef V51_RUNTIME_HPP
#define V51_RUNTIME_HPP

#include <json/json.h>

#include <string>

/* V51 runtime engine for processing and learning */
class V51RuntimeEngine {
public:
    V51RuntimeEngine();
    ~V51RuntimeEngine();

    V51RuntimeEngine(const V51RuntimeEngine &) = delete;
    V51RuntimeEngine &operator=(const V51RuntimeEngine &) = delete;

    /* Process request */
    Json::Value process(const Json::Value &request);
    /* Learn from request */
    Json::Value learn(const Json::Value &request);
    /* Get status for session */
    Json::Value status(const std::string &sessionId) const;

private:
    struct Impl;        /* Implementation */
    Impl *impl_;        /* Implementation pointer */
};

#endif // V51_RUNTIME_HPP
