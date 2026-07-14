/* v51_runtime.hpp - V51 runtime engine interface
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   079 Project is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>. */

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
