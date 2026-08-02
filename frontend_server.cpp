/* frontend_server.cpp - Frontend server implementation
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

#include <drogon/drogon.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cctype>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <deque>
#include <optional>
#include <random>
#include <unordered_set>
#include <numeric>
#include <cstring>
#include <cstdlib>
#include <type_traits>
#include "frontend_server.hpp"
#include "phoenix_config.hpp"
#include "emotion_system.hpp"
#include "mechanical_mind.hpp"
#include "transformer.hpp"
#include "loggerCXXH.hpp"
#include <jwt-cpp/jwt.h>
#include <jwt-cpp/traits/nlohmann-json/defaults.h>
#include <unordered_map>
#include <mutex>
#include <iomanip>
#include <atomic>
#include <thread>
#include <future>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <Eigen/Dense>

#include "DATABASE_079.hpp"
#include "speak_io.hpp"
#include "v51_runtime.hpp"
#include "gguf_tensor_parser.hpp"
#include "physics_world.hpp"
#include "physics_world_runtime.hpp"
#include "world_model.hpp"
#include "edge_platform.hpp"
#include "jpea_v2_image_world_model.hpp"

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

#ifdef HAVE_TORCH
#include <torch/torch.h>
#endif

namespace fs = std::filesystem;

using phoenix::resolveConfig;

namespace
{
    // 获取当前 Unix 毫秒时间戳。
    // 调用方式：用于记录用户创建时间、令牌过期时间等。
    // 实现思路：system_clock 转毫秒并返回 int64。
    // 注意事项：仅用于业务时间戳记录，不保证跨机时钟一致。
    static int64_t nowEpochMs()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    static void sendAuthErrorJson(const std::function<void(const drogon::HttpResponsePtr &)> &cb,
                                  drogon::HttpStatusCode code,
                                  const std::string &error,
                                  const std::string &message = std::string())
    {
        Json::Value out;
        out["ok"] = false;
        out["error"] = error;
        if (!message.empty())
            out["message"] = message;
        auto resp = drogon::HttpResponse::newHttpJsonResponse(out);
        resp->setStatusCode(code);
        cb(resp);
    }

    static nlohmann::json jsonCppToNlohmann(const Json::Value &value)
    {
        switch (value.type())
        {
        case Json::nullValue:
            return nullptr;
        case Json::intValue:
            return value.asInt64();
        case Json::uintValue:
            return value.asUInt64();
        case Json::realValue:
            return value.asDouble();
        case Json::stringValue:
            return value.asString();
        case Json::booleanValue:
            return value.asBool();
        case Json::arrayValue:
        {
            nlohmann::json out = nlohmann::json::array();
            for (const auto &entry : value)
            {
                out.push_back(jsonCppToNlohmann(entry));
            }
            return out;
        }
        case Json::objectValue:
        {
            nlohmann::json out = nlohmann::json::object();
            for (const auto &name : value.getMemberNames())
            {
                out[name] = jsonCppToNlohmann(value[name]);
            }
            return out;
        }
        default:
            return nullptr;
        }
    }

    static Json::Value nlohmannToJsonCpp(const nlohmann::json &value)
    {
        if (value.is_null())
        {
            return Json::Value(Json::nullValue);
        }
        if (value.is_boolean())
        {
            return Json::Value(value.get<bool>());
        }
        if (value.is_number_integer())
        {
            return Json::Value(static_cast<Json::Int64>(value.get<std::int64_t>()));
        }
        if (value.is_number_unsigned())
        {
            return Json::Value(static_cast<Json::UInt64>(value.get<std::uint64_t>()));
        }
        if (value.is_number_float())
        {
            return Json::Value(value.get<double>());
        }
        if (value.is_string())
        {
            return Json::Value(value.get<std::string>());
        }
        if (value.is_array())
        {
            Json::Value out(Json::arrayValue);
            for (const auto &entry : value)
            {
                out.append(nlohmannToJsonCpp(entry));
            }
            return out;
        }
        Json::Value out(Json::objectValue);
        for (auto it = value.begin(); it != value.end(); ++it)
        {
            out[it.key()] = nlohmannToJsonCpp(it.value());
        }
        return out;
    }

    static void sendNlohmannJson(const std::function<void(const drogon::HttpResponsePtr &)> &cb,
                                 const nlohmann::json &payload,
                                 drogon::HttpStatusCode code = drogon::k200OK)
    {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(code);
        resp->setContentTypeString("application/json");
        resp->setBody(payload.dump(2));
        cb(resp);
    }

    static nlohmann::json loadNlohmannJsonFile(const fs::path &path)
    {
        std::ifstream in(path);
        if (!in)
        {
            throw std::runtime_error("unable to open JSON file: " + path.string());
        }
        return nlohmann::json::parse(in, nullptr, true, true);
    }

    struct LocalAuthSession
    {
        std::string username;
        int64_t expMs{0};
    };

    static std::mutex gLocalAuthMu;
    static std::unordered_map<std::string, LocalAuthSession> gLocalAuthSessions;
    static std::atomic<uint64_t> gLocalAuthCounter{0};

    static std::string issueLocalAuthToken(const std::string &username, int64_t ttlMs)
    {
        uint64_t seq = ++gLocalAuthCounter;
        std::string token = "local-" + username + "-" + std::to_string(nowEpochMs()) + "-" + std::to_string(seq);
        std::lock_guard<std::mutex> lock(gLocalAuthMu);
        gLocalAuthSessions[token] = LocalAuthSession{username, nowEpochMs() + ttlMs};
        return token;
    }

    static bool parseLocalAuthToken(const std::string &token, std::string &usernameOut)
    {
        std::lock_guard<std::mutex> lock(gLocalAuthMu);
        auto it = gLocalAuthSessions.find(token);
        if (it == gLocalAuthSessions.end())
            return false;
        if (it->second.expMs > 0 && it->second.expMs < nowEpochMs())
        {
            gLocalAuthSessions.erase(it);
            return false;
        }
        usernameOut = it->second.username;
        return !usernameOut.empty();
    }

    static bool revokeLocalAuthToken(const std::string &token)
    {
        std::lock_guard<std::mutex> lock(gLocalAuthMu);
        return gLocalAuthSessions.erase(token) > 0;
    }

    static std::string extractBearerToken(const drogon::HttpRequestPtr &req)
    {
        auto auth = req->getHeader("authorization");
        const std::string prefix = "Bearer ";
        if (auth.rfind(prefix, 0) != 0)
            return {};
        return auth.substr(prefix.size());
    }

    // FrontendReservedArena 预留前端服务计算内存，减少运行期抖动。
    // 调用方式：由 ensureFrontendArena() 在首次请求前初始化。
    // 实现思路：按配置大小创建 byte 向量并触页。
    // 注意事项：此机制用于稳定性，不改变业务数据语义。
    // 注意事项：初始化具有幂等性，重复调用无副作用。
    // 注意事项：异常时清空数据，避免部分初始化状态。
    class FrontendReservedArena
    {
    public:
        // 初始化预留内存。
        // 调用方式：传入目标字节数，通常由统一入口调用。
        // 实现思路：做上下限裁剪后分配并逐页触达。
        // 注意事项：建议在启动或低峰时触发，避免阻塞请求。
        void init(size_t bytes)
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (inited_)
                return;
            bytes = std::max<size_t>(8ull * 1024ull * 1024ull, bytes);
            bytes = std::min<size_t>(1024ull * 1024ull * 1024ull, bytes);
            try
            {
                data_.resize(bytes);
                const size_t page = 4096;
                for (size_t i = 0; i < data_.size(); i += page)
                    data_[i] = 0;
            }
            catch (...)
            {
                data_.clear();
            }
            inited_ = true;
        }

    private:
        std::vector<uint8_t> data_;
        std::mutex mu_;
        bool inited_{false};
    };

    // 返回前端预留内存单例。
    // 调用方式：内部初始化流程统一通过本函数获取。
    // 实现思路：函数静态对象实现线程安全懒加载。
    // 注意事项：引用在进程生命周期内有效。
    static FrontendReservedArena &frontendArena()
    {
        static FrontendReservedArena arena;
        return arena;
    }

    // 确保前端预留内存仅初始化一次。
    // 调用方式：在服务入口或关键模块启动时调用。
    // 实现思路：读取 AI_FRONTEND_RESERVED_MB 并执行 init。
    // 注意事项：配置非法时回退默认值 96MB。
    static void ensureFrontendArena()
    {
        static std::once_flag once;
        std::call_once(once, []()
                       {
            double mb = resolveConfig<double>("frontend_server.reservedMemMb", 96.0, "AI_FRONTEND_RESERVED_MB");
            if (!std::isfinite(mb) || mb < 8.0)
                mb = 96.0;
            frontendArena().init((size_t)(mb * 1024.0 * 1024.0)); });
    }

            static std::atomic<int> gChatProxyInFlight{0};
            static std::mutex gChatDispatchMu;
            static std::chrono::steady_clock::time_point gLastChatDispatch = std::chrono::steady_clock::now();

    // UserRecord 表示用户账户的持久化字段。
    // 调用方式：由 UserStore 读写并在认证流程中传递。
    // 实现思路：聚合身份、凭据摘要、角色与令牌状态。
    // 注意事项：passwordHash/salt 为敏感字段，不应外泄。
    // 注意事项：过期时间采用毫秒时间戳。
    // 注意事项：结构体字段默认值确保未初始化也可安全序列化。
    struct UserRecord
    {
        std::string username;
        std::string email;
        std::string salt;
        std::string passwordHash;
        std::string role{"user"};
        int64_t createdAt{0};
        bool emailVerified{false};
        std::string emailVerifyTokenHash;
        int64_t emailVerifyTokenExp{0};
        std::string resetTokenHash;
        int64_t resetTokenExp{0};
    };

    static Json::Value userRecordToJson(const UserRecord &rec)
    {
        Json::Value out;
        out["username"] = rec.username;
        out["role"] = rec.role;
        out["email"] = rec.email;
        out["emailVerified"] = rec.emailVerified;
        out["createdAt"] = (Json::Int64)rec.createdAt;
        return out;
    }

    static bool hasAnyPrefix(const std::string &value, const std::vector<std::string> &prefixes)
    {
        for (const auto &prefix : prefixes)
        {
            if (!prefix.empty() && value.rfind(prefix, 0) == 0)
                return true;
        }
        return false;
    }

    // UserStore 负责用户注册、认证、找回密码与本地文件持久化。
    // 调用方式：构造时传入存储文件路径，后续调用各业务方法。
    // 实现思路：内存 map + 文件 JSON 快照，写操作后立即保存。
    // 注意事项：内部使用互斥锁保证并发安全。
    // 注意事项：该实现偏轻量，不替代专业用户中心。
    // 注意事项：调用方需自行做更严格的安全策略。
    class UserStore
    {
    public:
        // 构造用户存储并加载已有数据。
        // 调用方式：服务启动时传入用户数据文件路径。
        // 实现思路：保存路径后调用 load() 读取快照。
        // 注意事项：文件不存在时会以空库启动。
        explicit UserStore(fs::path filePath)
            : filePath_(std::move(filePath))
        {
            load();
        }

        // 判断是否已有用户。
        // 调用方式：用于初始化管理员或首用户流程。
        // 实现思路：在锁内检查 users_ 是否为空。
        // 注意事项：仅反映当前内存状态。
        bool hasUsers()
        {
            std::lock_guard<std::mutex> lock(mu_);
            return !users_.empty();
        }

        // 新增用户账号。
        // 调用方式：传入用户名/邮箱/密码/角色，失败写入 err。
        // 实现思路：先做校验与去重，再生成 salt 与哈希并落盘。
        // 注意事项：该接口要求最小密码长度，不含复杂度策略。
        bool addUser(const std::string &username, const std::string &email, const std::string &password, const std::string &role, std::string &err)
        {
            if (username.size() < 3 || username.size() > 32)
            {
                err = "invalid username";
                return false;
            }
            if (email.size() < 5 || email.find('@') == std::string::npos)
            {
                err = "invalid email";
                return false;
            }
            if (password.size() < 6)
            {
                err = "password too short";
                return false;
            }
            std::lock_guard<std::mutex> lock(mu_);
            if (users_.count(username))
            {
                err = "username exists";
                return false;
            }
            for (const auto &kv : users_)
            {
                if (kv.second.email == email)
                {
                    err = "email exists";
                    return false;
                }
            }
            UserRecord rec;
            rec.username = username;
            rec.email = email;
            rec.salt = randomHex(16);
            rec.passwordHash = hashPassword(password, rec.salt);
            rec.role = role;
            rec.createdAt = nowEpochMs();
            rec.emailVerified = false;
            users_[username] = rec;
            save();
            return true;
        }

        // 校验用户名密码并返回用户信息。
        // 调用方式：登录流程中调用，成功时 out 输出用户记录。
        // 实现思路：按用户名查找并做常规哈希对比。
        // 注意事项：仅用于本地实现，生产建议接入更强安全方案。
        bool verifyUser(const std::string &username, const std::string &password, UserRecord &out)
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = users_.find(username);
            if (it == users_.end())
                return false;
            std::string hashed = hashPassword(password, it->second.salt);
            if (!secureEquals(hashed, it->second.passwordHash))
                return false;
            out = it->second;
            return true;
        }

        // 按用户名获取用户。
        // 调用方式：管理接口或鉴权后补全资料时使用。
        // 实现思路：在锁内按 key 查询并复制记录。
        // 注意事项：未命中返回 false。
        bool getUser(const std::string &username, UserRecord &out)
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = users_.find(username);
            if (it == users_.end())
                return false;
            out = it->second;
            return true;
        }

        // 按邮箱获取用户。
        // 调用方式：找回密码与邮箱校验流程中使用。
        // 实现思路：遍历 users_ 比较 email 字段。
        // 注意事项：邮箱唯一性依赖 addUser 阶段约束。
        bool getUserByEmail(const std::string &email, UserRecord &out)
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (const auto &kv : users_)
            {
                if (kv.second.email == email)
                {
                    out = kv.second;
                    return true;
                }
            }
            return false;
        }

        // 覆盖更新用户记录并持久化。
        // 调用方式：修改用户状态、令牌或密码后调用。
        // 实现思路：按用户名定位后整体替换并 save。
        // 注意事项：调用方应保证 rec.username 不变且合法。
        bool updateUser(const UserRecord &rec)
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = users_.find(rec.username);
            if (it == users_.end())
                return false;
            it->second = rec;
            save();
            return true;
        }

        bool updateEmail(const std::string &username, const std::string &email, bool verified, std::string &err)
        {
            if (email.size() < 5 || email.find('@') == std::string::npos)
            {
                err = "invalid email";
                return false;
            }
            std::lock_guard<std::mutex> lock(mu_);
            auto it = users_.find(username);
            if (it == users_.end())
            {
                err = "user not found";
                return false;
            }
            for (const auto &kv : users_)
            {
                if (kv.first != username && kv.second.email == email)
                {
                    err = "email exists";
                    return false;
                }
            }
            if (it->second.email == email)
            {
                return true;
            }
            it->second.email = email;
            it->second.emailVerified = verified;
            it->second.emailVerifyTokenHash.clear();
            it->second.emailVerifyTokenExp = 0;
            save();
            return true;
        }

        bool changePassword(const std::string &username, const std::string &oldPassword, const std::string &newPassword, std::string &err)
        {
            if (newPassword.size() < 6)
            {
                err = "password too short";
                return false;
            }
            std::lock_guard<std::mutex> lock(mu_);
            auto it = users_.find(username);
            if (it == users_.end())
            {
                err = "user not found";
                return false;
            }
            std::string oldHashed = hashPassword(oldPassword, it->second.salt);
            if (!secureEquals(oldHashed, it->second.passwordHash))
            {
                err = "invalid credentials";
                return false;
            }
            std::string newHashed = hashPassword(newPassword, it->second.salt);
            if (secureEquals(newHashed, it->second.passwordHash))
            {
                err = "new password must differ";
                return false;
            }
            it->second.passwordHash = newHashed;
            it->second.resetTokenHash.clear();
            it->second.resetTokenExp = 0;
            save();
            return true;
        }

        Json::Value listUsersSummary(int limit, bool includeTestUsers)
        {
            std::lock_guard<std::mutex> lock(mu_);
            Json::Value users(Json::arrayValue);
            int emitted = 0;
            const std::vector<std::string> testPrefixes = {"autotest_", "bench_"};
            for (const auto &kv : users_)
            {
                const auto &rec = kv.second;
                bool testUser = hasAnyPrefix(rec.username, testPrefixes) || hasAnyPrefix(rec.email, testPrefixes);
                if (!includeTestUsers && testUser)
                    continue;
                users.append(userRecordToJson(rec));
                ++emitted;
                if (limit > 0 && emitted >= limit)
                    break;
            }
            return users;
        }

        size_t cleanupUsersByPrefixes(const std::vector<std::string> &prefixes, bool dryRun, Json::Value &removedOut)
        {
            std::lock_guard<std::mutex> lock(mu_);
            std::vector<std::string> toRemove;
            removedOut = Json::Value(Json::arrayValue);
            for (const auto &kv : users_)
            {
                const auto &rec = kv.second;
                bool matched = hasAnyPrefix(rec.username, prefixes) || hasAnyPrefix(rec.email, prefixes);
                if (!matched)
                    continue;
                removedOut.append(userRecordToJson(rec));
                toRemove.push_back(kv.first);
            }
            if (!dryRun)
            {
                for (const auto &name : toRemove)
                    users_.erase(name);
                if (!toRemove.empty())
                    save();
            }
            return toRemove.size();
        }

        // 生成邮箱验证令牌。
        // 调用方式：注册后或重发验证邮件时调用。
        // 实现思路：随机 token 经 hashToken 存储，并记录过期时间。
        // 注意事项：返回给用户的是明文 tokenOut，存储为哈希。
        bool issueEmailVerifyToken(const std::string &username, std::string &tokenOut, int64_t ttlMs)
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = users_.find(username);
            if (it == users_.end())
                return false;
            tokenOut = randomHex(20);
            it->second.emailVerifyTokenHash = hashToken(tokenOut, it->second.salt);
            it->second.emailVerifyTokenExp = nowEpochMs() + ttlMs;
            save();
            return true;
        }

        // 校验邮箱验证令牌。
        // 调用方式：用户点击验证链接后调用。
        // 实现思路：检查时效并做恒时比较，通过后清空令牌字段。
        // 注意事项：超时或错误 token 都返回 false。
        bool verifyEmailToken(const std::string &username, const std::string &token)
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = users_.find(username);
            if (it == users_.end())
                return false;
            if (it->second.emailVerifyTokenExp <= nowEpochMs())
                return false;
            if (!secureEquals(hashToken(token, it->second.salt), it->second.emailVerifyTokenHash))
                return false;
            it->second.emailVerified = true;
            it->second.emailVerifyTokenHash.clear();
            it->second.emailVerifyTokenExp = 0;
            save();
            return true;
        }

        // 根据邮箱签发重置密码令牌。
        // 调用方式：忘记密码流程中按邮箱触发。
        // 实现思路：遍历匹配邮箱后设置 resetTokenHash 与过期时间。
        // 注意事项：是否存在该邮箱建议在外层统一做防枚举处理。
        bool issueResetTokenByEmail(const std::string &email, std::string &tokenOut, int64_t ttlMs)
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (auto &kv : users_)
            {
                if (kv.second.email != email)
                    continue;
                tokenOut = randomHex(20);
                kv.second.resetTokenHash = hashToken(tokenOut, kv.second.salt);
                kv.second.resetTokenExp = nowEpochMs() + ttlMs;
                save();
                return true;
            }
            return false;
        }

        // 使用邮箱+令牌重置密码。
        // 调用方式：用户提交重置凭据时调用。
        // 实现思路：验证 token 与时效后更新密码哈希并清空令牌。
        // 注意事项：新密码仍需满足最小长度限制。
        bool resetPasswordByEmail(const std::string &email, const std::string &token, const std::string &newPassword)
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (auto &kv : users_)
            {
                auto &rec = kv.second;
                if (rec.email != email)
                    continue;
                if (rec.resetTokenExp <= nowEpochMs())
                    return false;
                if (!secureEquals(hashToken(token, rec.salt), rec.resetTokenHash))
                    return false;
                if (newPassword.size() < 6)
                    return false;
                rec.passwordHash = hashPassword(newPassword, rec.salt);
                rec.resetTokenHash.clear();
                rec.resetTokenExp = 0;
                save();
                return true;
            }
            return false;
        }

    private:
        // 从 JSON 文件加载用户数据。
        // 调用方式：构造阶段自动执行。
        // 实现思路：读取 root.users 数组并恢复到 users_。
        // 注意事项：格式异常会被安全忽略，保持可启动。
        void load()
        {
            std::lock_guard<std::mutex> lock(mu_);
            users_.clear();
            if (!fs::exists(filePath_))
                return;
            std::ifstream in(filePath_);
            if (!in)
                return;
            Json::Value root;
            in >> root;
            if (!root.isMember("users") || !root["users"].isArray())
                return;
            for (auto &u : root["users"])
            {
                if (!u.isObject())
                    continue;
                UserRecord rec;
                rec.username = u.get("username", "").asString();
                rec.email = u.get("email", "").asString();
                rec.salt = u.get("salt", "").asString();
                rec.passwordHash = u.get("passwordHash", "").asString();
                rec.role = u.get("role", "user").asString();
                rec.createdAt = u.get("createdAt", 0).asInt64();
                rec.emailVerified = u.get("emailVerified", false).asBool();
                rec.emailVerifyTokenHash = u.get("emailVerifyTokenHash", "").asString();
                rec.emailVerifyTokenExp = u.get("emailVerifyTokenExp", 0).asInt64();
                rec.resetTokenHash = u.get("resetTokenHash", "").asString();
                rec.resetTokenExp = u.get("resetTokenExp", 0).asInt64();
                if (!rec.username.empty() && !rec.passwordHash.empty())
                {
                    users_[rec.username] = rec;
                }
            }
        }

        // 将当前用户状态保存到 JSON 文件。
        // 调用方式：每次写操作后调用。
        // 实现思路：组装 users 数组并覆盖写入目标文件。
        // 注意事项：写文件失败未抛异常，建议外部监控磁盘状态。
        void save()
        {
            fs::create_directories(filePath_.parent_path());
            Json::Value root;
            Json::Value arr(Json::arrayValue);
            for (auto &kv : users_)
            {
                const auto &rec = kv.second;
                Json::Value u;
                u["username"] = rec.username;
                u["email"] = rec.email;
                u["salt"] = rec.salt;
                u["passwordHash"] = rec.passwordHash;
                u["role"] = rec.role;
                u["createdAt"] = (Json::Int64)rec.createdAt;
                u["emailVerified"] = rec.emailVerified;
                u["emailVerifyTokenHash"] = rec.emailVerifyTokenHash;
                u["emailVerifyTokenExp"] = (Json::Int64)rec.emailVerifyTokenExp;
                u["resetTokenHash"] = rec.resetTokenHash;
                u["resetTokenExp"] = (Json::Int64)rec.resetTokenExp;
                arr.append(u);
            }
            root["users"] = arr;
            std::ofstream out(filePath_);
            out << root.toStyledString();
        }

        // 生成十六进制随机串。
        // 调用方式：用于 salt 与一次性 token 生成。
        // 实现思路：随机字节逐个转 2 位 hex 拼接。
        // 注意事项：该实现依赖 std::random_device 可用性。
        static std::string randomHex(size_t bytes)
        {
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_int_distribution<int> dist(0, 255);
            std::ostringstream oss;
            for (size_t i = 0; i < bytes; i++)
            {
                int v = dist(rng);
                oss << std::hex << std::setw(2) << std::setfill('0') << v;
            }
            return oss.str();
        }

        // 计算密码哈希。
        // 调用方式：注册、登录校验和重置密码时调用。
        // 实现思路：将 password 与 salt 混合做多轮哈希扰动。
        // 注意事项：该方案为轻量实现，不等同 bcrypt/argon2。
        static std::string hashPassword(const std::string &password, const std::string &salt)
        {
            std::string data = password + ":" + salt;
            size_t h = 0x9e3779b97f4a7c15ull;
            for (int i = 0; i < 20000; i++)
            {
                h ^= std::hash<std::string>{}(data + std::to_string(h));
                h = (h << 13) ^ (h >> 7) ^ (h << 17);
            }
            std::ostringstream oss;
            oss << std::hex << h;
            return oss.str();
        }

        // 计算令牌哈希。
        // 调用方式：邮箱验证与重置密码令牌存储时调用。
        // 实现思路：复用 hashPassword 逻辑保证处理一致。
        // 注意事项：仅存储哈希值，不保存明文 token。
        static std::string hashToken(const std::string &token, const std::string &salt)
        {
            return hashPassword(token, salt);
        }

        // 执行恒时字符串比较。
        // 调用方式：比较密码哈希或 token 哈希。
        // 实现思路：逐字节异或累计差异，避免提前返回。
        // 注意事项：长度不同时直接返回 false。
        static bool secureEquals(const std::string &a, const std::string &b)
        {
            if (a.size() != b.size())
                return false;
            unsigned char diff = 0;
            for (size_t i = 0; i < a.size(); i++)
                diff |= (unsigned char)(a[i] ^ b[i]);
            return diff == 0;
        }

        fs::path filePath_;
        std::unordered_map<std::string, UserRecord> users_;
        std::mutex mu_;
    };

    std::mutex dbMutex;
    const std::string jwtSecret = "dev-secret-change-me";

    // Tokenizer 提供基础分词工具。
    // 调用方式：文本嵌入、上下文分析等流程调用 tokenize。
    // 实现思路：按单词字符聚合并做小写归一。
    // 注意事项：该分词器偏简单，适合轻量场景。
    // 注意事项：对多语言复杂分词支持有限。
    // 注意事项：UTF-8 高位字节按词字符处理。
    struct Tokenizer
    {
        // 判断字符是否属于词元。
        // 调用方式：tokenize 内部逐字符调用。
        // 实现思路：字母数字、下划线和高位字节视为词字符。
        // 注意事项：输入为 unsigned char 以避免符号扩展问题。
        static bool isWordChar(unsigned char c)
        {
            return std::isalnum(c) || c == '_' || c >= 0x80;
        }

        // 将原始文本拆分为词元列表。
        // 调用方式：传入文本字符串，返回词元向量。
        // 实现思路：连续词字符聚合为 token，遇分隔符切分。
        // 注意事项：输出 token 均转换为小写。
        static std::vector<std::string> tokenize(const std::string &text)
        {
            std::vector<std::string> out;
            std::string cur;
            for (size_t i = 0; i < text.size(); i++)
            {
                unsigned char c = static_cast<unsigned char>(text[i]);
                if (isWordChar(c))
                {
                    cur.push_back(static_cast<char>(std::tolower(c)));
                }
                else
                {
                    if (!cur.empty())
                    {
                        out.push_back(cur);
                        cur.clear();
                    }
                }
            }
            if (!cur.empty())
            {
                out.push_back(cur);
            }
            return out;
        }
    };

    static float sigmoid(float x);

    // EmbeddingStore 负责从语料构建词向量并提供文本向量化接口。
    // 调用方式：先 setCorpus，再通过 embedText 获取句向量。
    // 实现思路：统计共现矩阵→PPMI→SVD 得到词向量。
    // 注意事项：首次调用会触发一次性训练，可能较耗时。
    // 注意事项：内部使用 once_flag 保证只构建一次。
    // 注意事项：语料规模由 maxFiles/maxVocab 等参数控制。
    struct EmbeddingStore
    {
        int dim{128};
        int minCount{2};
        int window{4};
        size_t maxFiles{400};
        int maxVocab{3000};
        fs::path robotsDir;
        std::unordered_map<std::string, int> wordToId;
        std::vector<std::string> idToWord;
        std::unordered_map<std::string, int> docFreq;
        int docCount{0};
        std::vector<float> inEmb;
        bool ready{false};
        std::once_flag readyOnce;
        std::mutex mu;

        // 设置语料目录与最大文件数。
        // 调用方式：系统初始化时调用。
        // 实现思路：仅更新路径和上限参数，不立即训练。
        // 注意事项：目录应包含可读取 txt 文件。
        void setCorpus(const fs::path &dir, size_t maxFilesIn = 400)
        {
            robotsDir = dir;
            maxFiles = maxFilesIn;
        }

        // 将文本编码为固定维度向量。
        // 调用方式：输入文本，返回归一化句向量。
        // 实现思路：token 向量按 IDF 加权平均后归一化。
        // 注意事项：未知词会被跳过，可能导致全零向量。
        std::vector<float> embedText(const std::string &text)
        {
            ensureReady();
            auto tokens = Tokenizer::tokenize(text);
            std::vector<float> out(dim, 0.0f);
            if (tokens.empty())
                return out;
            double weightSum = 0.0;
            for (const auto &t : tokens)
            {
                float w = idf(t);
                auto v = getVector(t);
                if (v.empty())
                    continue;
                for (int i = 0; i < dim; i++)
                    out[i] += v[(size_t)i] * w;
                weightSum += w;
            }
            if (weightSum > 0)
            {
                for (int i = 0; i < dim; i++)
                    out[i] = (float)(out[i] / weightSum);
            }
            normalize(out);
            return out;
        }

        std::vector<float> get(const std::string &token)
        {
            ensureReady();
            auto out = getVector(token);
            if (out.empty())
                return std::vector<float>(dim, 0.0f);
            normalize(out);
            return out;
        }

    private:
        // 计算词的逆文档频率权重。
        // 调用方式：embedText 在聚合 token 向量前调用。
        // 实现思路：使用平滑公式 log(1 + docs/(1+df))。
        // 注意事项：当词未出现时会得到较高权重。
        float idf(const std::string &token) const
        {
            auto it = docFreq.find(token);
            int df = (it == docFreq.end()) ? 0 : it->second;
            int docs = std::max(1, docCount);
            return (float)std::log(1.0 + (double)docs / (1.0 + df));
        }

        // 读取单词对应的嵌入向量。
        // 调用方式：embedText 对每个 token 调用一次。
        // 实现思路：在锁内查词表并从连续内存切片拷贝。
        // 注意事项：未登录词返回空向量供上层跳过。
        std::vector<float> getVector(const std::string &token)
        {
            std::lock_guard<std::mutex> lock(mu);
            auto it = wordToId.find(token);
            if (it == wordToId.end())
                return {};
            int id = it->second;
            std::vector<float> out(dim, 0.0f);
            const size_t base = static_cast<size_t>(id) * (size_t)dim;
            for (int i = 0; i < dim; i++)
                out[i] = inEmb[base + (size_t)i];
            return out;
        }

        // 确保嵌入训练仅执行一次。
        // 调用方式：所有对外 embed 入口都会先调用。
        // 实现思路：通过 call_once 延迟触发 buildAndTrain。
        // 注意事项：首次调用可能有明显冷启动耗时。
        void ensureReady()
        {
            std::call_once(readyOnce, [this]()
                           {
                buildAndTrain();
                ready = true; });
        }

        // 构建词表并训练词向量矩阵。
        // 调用方式：由 ensureReady 的一次性初始化触发。
        // 实现思路：文本读取→共现统计→PPMI→SVD。
        // 注意事项：若语料不足会提前返回并保持默认状态。
        void buildAndTrain()
        {
            std::vector<std::vector<std::string>> corpus;
            if (fs::exists(robotsDir))
            {
                std::vector<fs::path> files;
                for (const auto &entry : fs::directory_iterator(robotsDir))
                {
                    if (!entry.is_regular_file())
                        continue;
                    if (entry.path().extension() != ".txt")
                        continue;
                    files.push_back(entry.path());
                }
                std::sort(files.begin(), files.end());
                size_t fileCount = 0;
                for (const auto &path : files)
                {
                    std::ifstream in(path, std::ios::binary);
                    if (!in)
                        continue;
                    std::ostringstream ss;
                    ss << in.rdbuf();
                    auto tokens = Tokenizer::tokenize(ss.str());
                    if (!tokens.empty())
                    {
                        corpus.push_back(tokens);
                        std::unordered_set<std::string> seen(tokens.begin(), tokens.end());
                        for (const auto &t : seen)
                            docFreq[t] += 1;
                        docCount += 1;
                    }
                    if (++fileCount >= maxFiles)
                        break;
                }
            }

            std::unordered_map<std::string, int> freq;
            for (const auto &doc : corpus)
            {
                for (const auto &t : doc)
                    freq[t] += 1;
            }

            std::vector<std::pair<std::string, int>> items;
            items.reserve(freq.size());
            for (const auto &kv : freq)
            {
                if (kv.second < minCount)
                    continue;
                items.push_back(kv);
            }
            std::sort(items.begin(), items.end(), [](const auto &a, const auto &b)
                      {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first; });
            if ((int)items.size() > maxVocab)
                items.resize((size_t)maxVocab);
            for (const auto &kv : items)
            {
                int id = (int)idToWord.size();
                wordToId[kv.first] = id;
                idToWord.push_back(kv.first);
            }

            if (idToWord.empty())
            {
                return;
            }

            if (dim > (int)idToWord.size())
                dim = (int)idToWord.size();

            const int V = (int)idToWord.size();
            Eigen::MatrixXf cooc = Eigen::MatrixXf::Zero(V, V);
            for (const auto &doc : corpus)
            {
                std::vector<int> ids;
                ids.reserve(doc.size());
                for (const auto &t : doc)
                {
                    auto it = wordToId.find(t);
                    if (it != wordToId.end())
                        ids.push_back(it->second);
                }
                if (ids.size() < 2)
                    continue;
                for (size_t i = 0; i < ids.size(); i++)
                {
                    int wi = ids[i];
                    size_t start = (i > (size_t)window) ? i - (size_t)window : 0;
                    size_t end = std::min(ids.size(), i + (size_t)window + 1);
                    for (size_t j = start; j < end; j++)
                    {
                        if (j == i)
                            continue;
                        int wj = ids[j];
                        float dist = (float)std::abs((int)j - (int)i);
                        float weight = dist > 0 ? 1.0f / dist : 1.0f;
                        cooc(wi, wj) += weight;
                    }
                }
            }

            Eigen::VectorXf rowSum = cooc.rowwise().sum();
            float total = rowSum.sum();
            if (total <= 0.0f)
                return;

            Eigen::MatrixXf ppmi(V, V);
            for (int i = 0; i < V; i++)
            {
                float ri = rowSum[i];
                for (int j = 0; j < V; j++)
                {
                    float x = cooc(i, j);
                    float rj = rowSum[j];
                    if (x <= 0.0f || ri <= 0.0f || rj <= 0.0f)
                    {
                        ppmi(i, j) = 0.0f;
                        continue;
                    }
                    float pmi = std::log((x * total) / (ri * rj));
                    ppmi(i, j) = pmi > 0.0f ? pmi : 0.0f;
                }
            }

            Eigen::BDCSVD<Eigen::MatrixXf> svd(ppmi, Eigen::ComputeThinU);
            Eigen::MatrixXf U = svd.matrixU().leftCols(dim);
            Eigen::VectorXf S = svd.singularValues().head(dim);

            inEmb.assign((size_t)V * (size_t)dim, 0.0f);
            for (int i = 0; i < V; i++)
            {
                float norm = 0.0f;
                for (int d = 0; d < dim; d++)
                {
                    float val = U(i, d) * std::sqrt(std::max(1e-9f, S[d]));
                    inEmb[(size_t)i * (size_t)dim + (size_t)d] = val;
                    norm += val * val;
                }
                norm = std::sqrt(std::max(1e-9f, norm));
                for (int d = 0; d < dim; d++)
                {
                    inEmb[(size_t)i * (size_t)dim + (size_t)d] /= norm;
                }
            }
        }

        // 对向量执行 L2 归一化。
        // 调用方式：句向量输出前及必要后处理路径调用。
        // 实现思路：计算平方和开方后逐维缩放。
        // 注意事项：使用最小范数下界避免除零。
        static void normalize(std::vector<float> &v)
        {
            double norm = 0.0;
            for (float x : v)
                norm += x * x;
            norm = std::sqrt(std::max(1e-9, norm));
            for (auto &x : v)
                x = (float)(x / norm);
        }
    };

    // Sigmoid 激活函数。
    // 调用方式：RNN/LSTM 门控计算中调用。
    // 实现思路：1/(1+exp(-x))。
    // 注意事项：极大正负值会受浮点范围影响。
    static float sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // tanh 快速封装。
    // 调用方式：循环网络状态更新时调用。
    // 实现思路：当前直接转调 std::tanh。
    // 注意事项：保留该包装便于后续替换近似实现。
    static float tanhFast(float x)
    {
        return std::tanh(x);
    }

    // RNNModel 表示简单循环神经网络单元。
    // 调用方式：构造后反复调用 step 推进时序状态。
    // 实现思路：h_t = tanh(Wx*x + Wh*h + b)。
    // 注意事项：该结构为推断侧轻量模型，不含训练逻辑。
    // 注意事项：输入维度与隐藏维度需与参数一致。
    // 注意事项：参数初始化使用均匀分布。
    struct RNNModel
    {
        int inputDim{128};
        int hiddenDim{128};
        std::vector<float> Wx;
        std::vector<float> Wh;
        std::vector<float> b;

        // 构造并随机初始化 RNN 参数。
        // 调用方式：传入输入维和隐藏维，不传则使用默认 128。
        // 实现思路：Wx/Wh/b 按小范围均匀分布初始化。
        // 注意事项：固定种子可提升可复现性。
        explicit RNNModel(int inDim = 128, int hidDim = 128)
            : inputDim(inDim), hiddenDim(hidDim),
              Wx(static_cast<size_t>(hidDim * inDim), 0.0f),
              Wh(static_cast<size_t>(hidDim * hidDim), 0.0f),
              b(static_cast<size_t>(hidDim), 0.0f)
        {
            std::mt19937 rng(7);
            std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
            for (auto &v : Wx)
                v = dist(rng);
            for (auto &v : Wh)
                v = dist(rng);
            for (auto &v : b)
                v = dist(rng);
        }

            // 执行单步 RNN 前向。
            // 调用方式：输入当前 x 与上一时刻 h，返回新 hidden。
            // 实现思路：逐维累加输入项与循环项后过 tanh。
            // 注意事项：调用方需保证 x/h 维度匹配。
        std::vector<float> step(const std::vector<float> &x, const std::vector<float> &h) const
        {
            std::vector<float> out(hiddenDim, 0.0f);
            for (int r = 0; r < hiddenDim; r++)
            {
                float sum = b[r];
                const size_t rowX = static_cast<size_t>(r) * inputDim;
                const size_t rowH = static_cast<size_t>(r) * hiddenDim;
                for (int c = 0; c < inputDim; c++)
                {
                    sum += Wx[rowX + c] * x[c];
                }
                for (int c = 0; c < hiddenDim; c++)
                {
                    sum += Wh[rowH + c] * h[c];
                }
                out[r] = tanhFast(sum);
            }
            return out;
        }
    };

    // LSTMModel 表示四门控的 LSTM 单元。
    // 调用方式：构造后按时间步调用 step 传递 h/c。
    // 实现思路：计算 f/i/o/g 四门并更新 cell 与 hidden。
    // 注意事项：该实现用于轻量推理，不包含反向传播。
    // 注意事项：参数按统一随机策略初始化。
    // 注意事项：输入状态维度需严格对齐 hiddenDim。
    struct LSTMModel
    {
        int inputDim{128};
        int hiddenDim{128};
        std::vector<float> Wf;
        std::vector<float> Wi;
        std::vector<float> Wo;
        std::vector<float> Wg;
        std::vector<float> Uf;
        std::vector<float> Ui;
        std::vector<float> Uo;
        std::vector<float> Ug;
        std::vector<float> bf;
        std::vector<float> bi;
        std::vector<float> bo;
        std::vector<float> bg;

        // 构造并初始化 LSTM 参数。
        // 调用方式：传入输入维和隐藏维。
        // 实现思路：四组输入权重、循环权重和偏置统一初始化。
        // 注意事项：默认维度为 128/128。
        explicit LSTMModel(int inDim = 128, int hidDim = 128)
            : inputDim(inDim), hiddenDim(hidDim),
              Wf(static_cast<size_t>(hidDim * inDim), 0.0f),
              Wi(static_cast<size_t>(hidDim * inDim), 0.0f),
              Wo(static_cast<size_t>(hidDim * inDim), 0.0f),
              Wg(static_cast<size_t>(hidDim * inDim), 0.0f),
              Uf(static_cast<size_t>(hidDim * hidDim), 0.0f),
              Ui(static_cast<size_t>(hidDim * hidDim), 0.0f),
              Uo(static_cast<size_t>(hidDim * hidDim), 0.0f),
              Ug(static_cast<size_t>(hidDim * hidDim), 0.0f),
              bf(static_cast<size_t>(hidDim), 0.0f),
              bi(static_cast<size_t>(hidDim), 0.0f),
              bo(static_cast<size_t>(hidDim), 0.0f),
              bg(static_cast<size_t>(hidDim), 0.0f)
        {
            std::mt19937 rng(17);
            std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
            auto init = [&](std::vector<float> &v)
            {
                for (auto &x : v)
                    x = dist(rng);
            };
            init(Wf);
            init(Wi);
            init(Wo);
            init(Wg);
            init(Uf);
            init(Ui);
            init(Uo);
            init(Ug);
            init(bf);
            init(bi);
            init(bo);
            init(bg);
        }

        // 执行单步 LSTM 前向。
        // 调用方式：输入 x、h、c，返回 nextH 与 nextC。
        // 实现思路：按标准 LSTM 方程计算四门并更新状态。
        // 注意事项：向量维度不一致会导致越界风险。
        std::pair<std::vector<float>, std::vector<float>> step(const std::vector<float> &x,
                                                               const std::vector<float> &h,
                                                               const std::vector<float> &c) const
        {
            std::vector<float> nextH(hiddenDim, 0.0f);
            std::vector<float> nextC(hiddenDim, 0.0f);
            for (int r = 0; r < hiddenDim; r++)
            {
                float f = bf[r];
                float i = bi[r];
                float o = bo[r];
                float g = bg[r];
                const size_t rowX = static_cast<size_t>(r) * inputDim;
                const size_t rowH = static_cast<size_t>(r) * hiddenDim;
                for (int cIdx = 0; cIdx < inputDim; cIdx++)
                {
                    f += Wf[rowX + cIdx] * x[cIdx];
                    i += Wi[rowX + cIdx] * x[cIdx];
                    o += Wo[rowX + cIdx] * x[cIdx];
                    g += Wg[rowX + cIdx] * x[cIdx];
                }
                for (int cIdx = 0; cIdx < hiddenDim; cIdx++)
                {
                    f += Uf[rowH + cIdx] * h[cIdx];
                    i += Ui[rowH + cIdx] * h[cIdx];
                    o += Uo[rowH + cIdx] * h[cIdx];
                    g += Ug[rowH + cIdx] * h[cIdx];
                }
                float fGate = sigmoid(f);
                float iGate = sigmoid(i);
                float oGate = sigmoid(o);
                float gGate = tanhFast(g);
                float cNew = fGate * c[r] + iGate * gGate;
                nextC[r] = cNew;
                nextH[r] = oGate * tanhFast(cNew);
            }
            return {nextH, nextC};
        }
    };

    // ShortTermLlamaWindow 维护最短期消息窗口，输出 llama-server 风格的消息串。
    // 调用方式：每轮用户消息调用 pushUser，组装提示时调用 render。
    // 实现思路：按消息数和 token 预算双阈值裁剪，模拟聊天消息窗口。
    // 注意事项：该结构只维护短窗口，不承担长期摘要。
    // 注意事项：线程安全由外层 Session 容器保证。
    struct ShortTermLlamaWindow
    {
        struct Message
        {
            std::string role;
            std::string content;
            size_t tokenCount{0};
        };

        std::deque<Message> messages;
        size_t maxMessages{40};
        size_t maxTokens{2048};

        ShortTermLlamaWindow()
        {
            // 运行时可调：FRONTEND_SHORT_WINDOW_MAX_MESSAGES / FRONTEND_SHORT_WINDOW_MAX_TOKENS
            // 默认值从 config/phoenix.json 读取；环境变量仍可覆盖。
            static size_t envMaxMessages = 40;
            static size_t envMaxTokens = 2048;
            static std::once_flag winInit;
            std::call_once(winInit, []() {
                envMaxMessages = std::max<size_t>(1, resolveConfig<size_t>("context.adaptive.shortWindowMaxMessages", 40, "FRONTEND_SHORT_WINDOW_MAX_MESSAGES"));
                envMaxTokens = std::max<size_t>(64, resolveConfig<size_t>("context.adaptive.shortWindowMaxTokens", 2048, "FRONTEND_SHORT_WINDOW_MAX_TOKENS"));
            });
            maxMessages = envMaxMessages;
            maxTokens = envMaxTokens;
        }

        void pushUser(const std::string &text)
        {
            pushRole("user", text);
        }

        // 将一条助手回复推入短期窗口。
        // 调用方式：每轮模型回复后调用，配合 pushUser 形成完整多轮历史。
        // 实现思路：与 pushUser 共用裁剪逻辑，role 标记为 assistant。
        // 注意事项：空回复会被忽略，避免污染窗口。
        void pushAssistant(const std::string &text)
        {
            pushRole("assistant", text);
        }

        void pushRole(const std::string &role, const std::string &text)
        {
            if (text.empty())
            {
                return;
            }
            Message item;
            item.role = role;
            item.content = text;
            item.tokenCount = std::max<size_t>(1, Tokenizer::tokenize(text).size());
            messages.push_back(std::move(item));

            size_t totalTokens = 0;
            for (const auto &entry : messages)
            {
                totalTokens += entry.tokenCount;
            }
            while (!messages.empty() && (messages.size() > maxMessages || totalTokens > maxTokens))
            {
                totalTokens -= messages.front().tokenCount;
                messages.pop_front();
            }
        }

        std::string render() const
        {
            std::ostringstream ss;
            for (const auto &entry : messages)
            {
                ss << "<|start_header_id|>" << entry.role << "<|end_header_id|>\n";
                ss << entry.content << "\n";
                ss << "<|eot_id|>\n";
            }
            return ss.str();
        }
    };

    // ConcatContext 维护传统上下文窗口，作为长期记忆回溯的词面支撑。
    // 调用方式：每轮消息调用 push，长期融合时调用 concat。
    // 实现思路：使用 deque 保存最近若干条文本并按顺序连接。
    // 注意事项：该结构与短期窗口并行存在，用于混合长期上下文。
    struct ConcatContext
    {
        std::deque<std::string> history;
        size_t maxItems{8};

        // 将一条文本推入上下文窗口。
        // 调用方式：每次用户或模型产生新文本时调用。
        // 实现思路：push_back 后按上限弹出最旧片段。
        // 注意事项：窗口大小由 maxItems 控制。
        void push(const std::string &text)
        {
            history.push_back(text);
            while (history.size() > maxItems)
            {
                history.pop_front();
            }
        }

        // 按时间顺序拼接当前窗口内容。
        // 调用方式：组装模型输入上下文时调用。
        // 实现思路：用换行符连接 deque 中各片段。
        // 注意事项：当历史为空时返回空字符串。
        std::string concat() const
        {
            std::ostringstream ss;
            for (size_t i = 0; i < history.size(); i++)
            {
                if (i > 0)
                {
                    ss << "\n";
                }
                ss << history[i];
            }
            return ss.str();
        }
    };

    // EpisodicMemoryEntry 存储跨 session 的对话摘要和关键事实（Episodic Memory 层）。
    // 调用方式：ContextService 在会话结束时自动归档，新会话启动时检索注入。
    // 实现思路：基于 embedding 的语义相似度检索，保留用户偏好、决策上下文等有用信息。
    // 注意事项：采用轻量级向量检索，避免引入外部向量数据库依赖。
    // 注意事项：支持时间衰减，近期事实权重更高。
    struct EpisodicMemoryEntry
    {
        std::string sessionId;        // 来源会话 ID
        std::string summary;          // 对话摘要（由 LLM 生成或规则提取）
        std::vector<std::string> facts; // 关键事实列表（如 "用户偏好中文回复"）
        std::vector<float> embedding;  // 摘要的 embedding 向量
        int64_t timestamp;            // 归档时间戳
        float relevanceScore;         // 相关性分数（检索时计算）

        // 计算与查询 embedding 的余弦相似度。
        // 调用方式：检索时调用，用于排序记忆条目。
        // 实现思路：点积 / (||a|| * ||b||)。
        // 注意事项：返回值范围 [-1, 1]，越高越相关。
        float cosineSimilarity(const std::vector<float> &queryEmbedding) const
        {
            if (embedding.empty() || queryEmbedding.empty() || embedding.size() != queryEmbedding.size())
                return 0.0f;
            float dot = 0.0f, normA = 0.0f, normB = 0.0f;
            for (size_t i = 0; i < embedding.size(); ++i)
            {
                dot += embedding[i] * queryEmbedding[i];
                normA += embedding[i] * embedding[i];
                normB += queryEmbedding[i] * queryEmbedding[i];
            }
            float norm = std::sqrt(normA) * std::sqrt(normB);
            return norm > 1e-9f ? dot / norm : 0.0f;
        }
    };

    // SessionState 保存单个会话的文本模型运行状态。
    // 调用方式：ContextService 通过 sessionId 在 map 中持有。
    // 实现思路：聚合 RNN/LSTM 隐状态、拼接上下文和计数器。
    // 注意事项：状态与会话绑定，不应跨用户复用。
    // 注意事项：字段默认值用于确保首次访问安全。
    // 注意事项：并发更新需通过外层互斥锁串行化。
    // AdaptiveController: 每个 session 独立的元参数自适应控制器。
    // 根据实测延迟动态调整 concatThresh/rnnThresh/maxMessages，
    // 在延迟过高时压缩上下文，在效果不足时扩展上下文。
    // 所有调整在 cooldown 期内冻结，防止震荡。
    struct AdaptiveController
    {
        int concatThresh{5};      // 当前 concat->rnn 切换轮次
        int rnnThresh{15};        // 当前 rnn->lstm 切换轮次
        int maxMessages{40};      // 当前 shortWindow 消息上限
        float avgLatencyMs{0.0f}; // EMA 延迟（α=0.2）
        int adjustCooldown{0};    // 调整冷却倒计时（轮次）
        int roundCount{0};        // 已处理轮次
        float targetLatencyMs{15000.0f}; // 目标延迟阈值（ms）

        AdaptiveController()
        {
            // 初始值从 config/phoenix.json 读取；环境变量仍可覆盖。
            concatThresh = std::max(1, resolveConfig<int>("context.adaptive.concatThresh", 5, "FRONTEND_CONCAT_THRESH"));
            rnnThresh = std::max(concatThresh + 1, resolveConfig<int>("context.adaptive.rnnThresh", 15, "FRONTEND_RNN_THRESH"));
            maxMessages = std::max(1, resolveConfig<int>("context.adaptive.shortWindowMaxMessages", 40, "FRONTEND_SHORT_WINDOW_MAX_MESSAGES"));
            targetLatencyMs = std::max(1000.0f, resolveConfig<float>("context.adaptive.targetLatencyMs", 15000.0f, "FRONTEND_TARGET_LATENCY_MS"));
        }

        // 每轮对话结束后调用，传入本轮实测延迟（毫秒）。
        void update(float latencyMs)
        {
            const float alpha = 0.2f;
            if (roundCount == 0)
                avgLatencyMs = latencyMs;
            else
                avgLatencyMs = alpha * latencyMs + (1.0f - alpha) * avgLatencyMs;
            roundCount++;
            if (adjustCooldown > 0)
            {
                adjustCooldown--;
                return;
            }
            // 每 10 轮评估一次
            if (roundCount % 10 != 0) return;
            adapt();
        }

        void adapt()
        {
            bool changed = false;
            if (avgLatencyMs > targetLatencyMs)
            {
                // 延迟过高：压缩上下文
                if (maxMessages > 5)
                {
                    maxMessages = std::max(5, maxMessages - 5);
                    changed = true;
                }
                if (rnnThresh > concatThresh + 2)
                {
                    rnnThresh = std::max(concatThresh + 2, rnnThresh - 3);
                    changed = true;
                }
            }
            else if (avgLatencyMs < targetLatencyMs * 0.5f)
            {
                // 延迟充裕：可以扩展上下文
                if (maxMessages < 40)
                {
                    maxMessages = std::min(40, maxMessages + 5);
                    changed = true;
                }
                if (rnnThresh < 25)
                {
                    rnnThresh = std::min(25, rnnThresh + 3);
                    changed = true;
                }
            }
            if (changed)
            {
                adjustCooldown = 20; // 冷却 20 轮
                std::cout << "[adaptive] session adjusted: concatThresh=" << concatThresh
                          << " rnnThresh=" << rnnThresh
                          << " maxMessages=" << maxMessages
                          << " avgLatencyMs=" << avgLatencyMs << std::endl;
            }
        }
    };

    struct SessionState
    {
        std::vector<float> rnnHidden;
        std::vector<float> lstmHidden;
        std::vector<float> lstmCell;
        ShortTermLlamaWindow shortWindow;
        ConcatContext concat;
        std::string lastMode;
        int messageCount{0};
        AdaptiveController adaptive; // 运行时自适应元参数控制器
    };

#ifdef HAVE_TORCH
    // TorchVisionPipeline 提供基于 Torch 的图像检测与分类流程。
    // 调用方式：服务启动后构建实例并调用 infer 获取结果。
    // 实现思路：封装 YOLO 与 CNN 子模型及训练/加载逻辑。
    // 注意事项：模型文件路径与参数由环境变量配置。
    // 注意事项：未准备就绪时应由调用方处理降级路径。
    // 注意事项：内部线程安全由成员互斥锁与调用顺序保证。
    struct TorchVisionPipeline
    {
        // DetBox 表示检测框的类别、置信度与归一化几何参数。
        // 调用方式：由检测头解码后写入 DetItem 或推理结果。
        // 实现思路：保存 cls/conf 和中心点+宽高参数。
        // 注意事项：数值范围依赖模型输出后处理策略。
        // 注意事项：坐标解释需与当前输入分辨率对应。
        // 注意事项：该结构为轻量值对象，按值复制成本低。
        struct DetBox
        {
            int cls{0};
            float conf{0.0f};
            float cx{0.0f};
            float cy{0.0f};
            float w{0.0f};
            float h{0.0f};
        };

        // YoloNetImpl 定义轻量目标检测网络骨干与输出头。
        // 调用方式：由 TorchVisionPipeline 构建并执行 forward。
        // 实现思路：卷积骨干提取特征，1x1 head 输出网格预测。
        // 注意事项：classes/S/B 需与训练标签和损失函数一致。
        // 注意事项：该模块仅定义网络，不包含 NMS 后处理。
        // 注意事项：参数初始化依赖 Torch 默认策略。
        struct YoloNetImpl : torch::nn::Module
        {
            int classes{1};
            int S{7};
            int B{2};
            torch::nn::Sequential backbone{nullptr};
            torch::nn::Conv2d head{nullptr};

            YoloNetImpl(int c, int s = 7, int b = 2) : classes(c), S(s), B(b)
            {
                backbone = torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1).padding(1)),
                    torch::nn::BatchNorm2d(16), torch::nn::ReLU(),
                    torch::nn::MaxPool2d(2),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1).padding(1)),
                    torch::nn::BatchNorm2d(32), torch::nn::ReLU(),
                    torch::nn::MaxPool2d(2),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
                    torch::nn::BatchNorm2d(64), torch::nn::ReLU(),
                    torch::nn::MaxPool2d(2),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
                    torch::nn::BatchNorm2d(128), torch::nn::ReLU(),
                    torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({S, S})));
                int outCh = B * 5 + classes;
                head = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, outCh, 1));
                register_module("backbone", backbone);
                register_module("head", head);
            }

            // 执行 YOLO 网络前向推理。
            // 调用方式：训练与在线推理阶段均会调用。
            // 实现思路：先过 backbone，再由 head 输出网格预测。
            // 注意事项：输出张量仍需外部完成解码与 NMS。
            torch::Tensor forward(torch::Tensor x)
            {
                x = backbone->forward(x);
                x = head->forward(x);
                return x;
            }
        };
        TORCH_MODULE(YoloNet);

        // CnnNetImpl 定义分类与嵌入提取的卷积网络。
        // 调用方式：推理时可调用 forwardWithEmbedding 获取双输出。
        // 实现思路：卷积特征抽取后经全连接得到 embedding 与 logits。
        // 注意事项：classes 与训练数据标签空间必须一致。
        // 注意事项：embedding 维度由 embDim 参数控制。
        // 注意事项：输入尺寸需与预处理步骤匹配。
        struct CnnNetImpl : torch::nn::Module
        {
            int classes{1};
            torch::nn::Sequential features{nullptr};
            torch::nn::Linear fc1{nullptr};
            torch::nn::Linear fc2{nullptr};

            CnnNetImpl(int c, int embDim = 128) : classes(c)
            {
                features = torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1).padding(1)),
                    torch::nn::BatchNorm2d(16), torch::nn::ReLU(),
                    torch::nn::MaxPool2d(2),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(1).padding(1)),
                    torch::nn::BatchNorm2d(32), torch::nn::ReLU(),
                    torch::nn::MaxPool2d(2),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
                    torch::nn::BatchNorm2d(64), torch::nn::ReLU(),
                    torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({4, 4})));
                fc1 = torch::nn::Linear(64 * 4 * 4, embDim);
                fc2 = torch::nn::Linear(embDim, classes);
                register_module("features", features);
                register_module("fc1", fc1);
                register_module("fc2", fc2);
            }

            // 返回分类 logits 与中间 embedding。
            // 调用方式：分类推理和表征提取共用此接口。
            // 实现思路：卷积特征展平后依次通过 fc1/fc2。
            // 注意事项：embedding 使用 ReLU 激活后的表示。
            std::pair<torch::Tensor, torch::Tensor> forwardWithEmbedding(torch::Tensor x)
            {
                x = features->forward(x);
                x = x.view({x.size(0), -1});
                auto emb = torch::relu(fc1->forward(x));
                auto logits = fc2->forward(emb);
                return {logits, emb};
            }
        };
        TORCH_MODULE(CnnNet);

        // ClsItem 表示分类训练样本的路径与标签索引。
        // 调用方式：加载数据集元信息后用于批次采样。
        // 实现思路：以文件路径 + 数值标签组织最小样本单元。
        // 注意事项：path 需指向可读取图像文件。
        // 注意事项：label 对应 clsClasses_ 的下标。
        // 注意事项：无效样本会在数据加载阶段被过滤。
        struct ClsItem
        {
            fs::path path;
            int label;
        };
        // DetItem 表示检测训练样本及其标注框集合。
        // 调用方式：检测数据集加载后由训练流程消费。
        // 实现思路：每张图片路径关联多个 DetBox 标注。
        // 注意事项：boxes 为空时样本通常应被跳过或特殊处理。
        // 注意事项：标注坐标格式需与训练目标编码一致。
        // 注意事项：该结构不负责标注文件解析。
        struct DetItem
        {
            fs::path path;
            std::vector<DetBox> boxes;
        };

        // 构造视觉流水线并加载配置与模型元信息。
        // 调用方式：服务初始化阶段创建单例时调用。
        // 实现思路：读取环境变量、创建目录、加载数据并初始化网络。
        // 注意事项：构造阶段失败不会抛出，调用方需检查 ready。
        TorchVisionPipeline()
        {
            dataRoot_ = fs::path(resolveConfig<std::string>("vision.dataRoot", std::string("/graphies"), "VISION_DATA_ROOT"));
            modelDir_ = fs::path(resolveConfig<std::string>("vision.modelDir", std::string("./models/vision"), "VISION_MODEL_DIR"));
            clsInput_ = std::max(64, resolveConfig<int>("vision.torch.cnnInput", 224, "VISION_TORCH_CNN_INPUT"));
            detInput_ = std::max(96, resolveConfig<int>("vision.torch.yoloInput", 416, "VISION_TORCH_YOLO_INPUT"));
            detS_ = std::max(4, resolveConfig<int>("vision.torch.yoloS", 7, "VISION_TORCH_YOLO_S"));
            detB_ = std::max(1, resolveConfig<int>("vision.torch.yoloB", 2, "VISION_TORCH_YOLO_B"));
            detScore_ = resolveConfig<float>("vision.torch.yoloScore", 0.25f, "VISION_TORCH_YOLO_SCORE");
            detNms_ = resolveConfig<float>("vision.torch.yoloNms", 0.45f, "VISION_TORCH_YOLO_NMS");
            detEpochs_ = std::max(1, resolveConfig<int>("vision.torch.yoloEpochs", 10, "VISION_TORCH_YOLO_EPOCHS"));
            clsEpochs_ = std::max(1, resolveConfig<int>("vision.torch.cnnEpochs", 8, "VISION_TORCH_CNN_EPOCHS"));
            batch_ = std::max(1, resolveConfig<int>("vision.torch.batch", 8, "VISION_TORCH_BATCH"));
            lr_ = resolveConfig<float>("vision.torch.lr", 0.001f, "VISION_TORCH_LR");
            forceTrain_ = resolveConfig<bool>("vision.torch.forceTrain", false, "VISION_TORCH_FORCE_TRAIN");

            if (!fs::exists(modelDir_))
            {
                std::error_code ec;
                fs::create_directories(modelDir_, ec);
            }
            loadDatasetMeta();
            initModels();
        }

        // 判断检测与分类模型是否都已可用。
        // 调用方式：推理前快速检查可运行状态。
        // 实现思路：组合 yoloReady_ 与 cnnReady_ 两个标志位。
        // 注意事项：ready=true 不代表权重已完成训练。
        bool ready() const
        {
            return yoloReady_ && cnnReady_;
        }

        // 确保模型已完成训练或成功加载权重。
        // 调用方式：analyze 开始时调用，保证推理参数可用。
        // 实现思路：加锁后按 forceTrain/权重存在性决定训练流程。
        // 注意事项：该函数可能耗时较长，不应放在超短超时路径。
        void ensureTrained()
        {
            std::lock_guard<std::mutex> lock(trainMu_);
            if (trained_)
                return;
            if (!fs::exists(dataRoot_))
            {
                return;
            }
            if (forceTrain_ || !loadWeights())
            {
                trainAll();
                saveWeights();
            }
            trained_ = true;
        }

        // 对输入图像执行检测与分类联合分析。
        // 调用方式：视觉 API 收到图像后调用并返回结构化结果。
        // 实现思路：先校验状态，再运行 YOLO 与 CNN 并汇总输出。
        // 注意事项：空图像或模型未就绪时会返回 error。
        Json::Value analyze(const cv::Mat &img)
        {
            ensureTrained();
            Json::Value out;
            out["ok"] = false;
            if (!ready())
            {
                out["error"] = "vision model not ready";
                return out;
            }
            if (img.empty())
            {
                out["error"] = "empty image";
                return out;
            }
            auto dets = inferYolo(img);
            auto detail = inferCnn(img);

            Json::Value detArr(Json::arrayValue);
            for (const auto &d : dets)
            {
                Json::Value item;
                item["label"] = className(d.cls, detClasses_);
                item["score"] = d.conf;
                Json::Value box(Json::arrayValue);
                box.append((int)std::round(d.cx - d.w * 0.5f));
                box.append((int)std::round(d.cy - d.h * 0.5f));
                box.append((int)std::round(d.w));
                box.append((int)std::round(d.h));
                item["box"] = box;
                detArr.append(item);
            }
            Json::Value detDetail(Json::arrayValue);
            for (const auto &p : detail.topk)
            {
                Json::Value item;
                item["label"] = p.first;
                item["score"] = p.second;
                detDetail.append(item);
            }
            Json::Value emb(Json::arrayValue);
            for (float v : detail.embedding)
                emb.append(v);

            out["ok"] = true;
            out["detections"] = detArr;
            out["details"] = detDetail;
            out["embedding"] = emb;
            out["embeddingDim"] = (int)detail.embedding.size();
            out["imageSize"] = Json::Value(Json::objectValue);
            out["imageSize"]["w"] = img.cols;
            out["imageSize"]["h"] = img.rows;
            out["graphContext"] = buildGraphContext(dets, detail.topk);
            return out;
        }

    private:
        // CnnDetail 保存分类推理的 topk 结果与向量嵌入。
        // 调用方式：由 runCnn 内部构建并传递给上层组装响应。
        // 实现思路：聚合标签分数对与 embedding 向量。
        // 注意事项：topk 项数量受调用参数或配置限制。
        // 注意事项：embedding 维度取决于模型 fc1 设定。
        // 注意事项：该结构仅用于一次推理上下文。
        struct CnnDetail
        {
            std::vector<std::pair<std::string, float>> topk;
            std::vector<float> embedding;
        };

        void loadDatasetMeta()
        {
            clsItems_.clear();
            clsClasses_.clear();
            detItems_.clear();
            detClasses_.clear();

            fs::path clsRoot = dataRoot_ / "cls";
            if (fs::exists(clsRoot))
            {
                for (auto &entry : fs::directory_iterator(clsRoot))
                {
                    if (!entry.is_directory())
                        continue;
                    std::string cname = entry.path().filename().string();
                    int label = (int)clsClasses_.size();
                    clsClasses_.push_back(cname);
                    for (auto &img : fs::recursive_directory_iterator(entry.path()))
                    {
                        if (!img.is_regular_file())
                            continue;
                        auto ext = img.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                        {
                            clsItems_.push_back({img.path(), label});
                        }
                    }
                }
            }

            fs::path detRoot = dataRoot_ / "det";
            fs::path detImg = detRoot / "images";
            fs::path detLbl = detRoot / "labels";
            fs::path detCls = detRoot / "classes.txt";
            if (fs::exists(detCls))
            {
                std::ifstream in(detCls);
                std::string line;
                while (std::getline(in, line))
                    if (!line.empty())
                        detClasses_.push_back(line);
            }
            if (fs::exists(detImg) && fs::exists(detLbl))
            {
                for (auto &entry : fs::directory_iterator(detImg))
                {
                    if (!entry.is_regular_file())
                        continue;
                    auto ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (!(ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp"))
                        continue;
                    auto stem = entry.path().stem().string();
                    fs::path lblPath = detLbl / (stem + ".txt");
                    if (!fs::exists(lblPath))
                        continue;
                    std::vector<DetBox> boxes;
                    std::ifstream in(lblPath);
                    std::string line;
                    while (std::getline(in, line))
                    {
                        std::stringstream ss(line);
                        int cls = 0;
                        float cx = 0, cy = 0, w = 0, h = 0;
                        ss >> cls >> cx >> cy >> w >> h;
                        DetBox b;
                        b.cls = cls;
                        b.cx = cx;
                        b.cy = cy;
                        b.w = w;
                        b.h = h;
                        b.conf = 1.0f;
                        boxes.push_back(b);
                        if (cls >= (int)detClasses_.size())
                        {
                            detClasses_.resize((size_t)cls + 1);
                            detClasses_[cls] = "class_" + std::to_string(cls);
                        }
                    }
                    if (!boxes.empty())
                        detItems_.push_back({entry.path(), boxes});
                }
            }
            if (detClasses_.empty())
                detClasses_.push_back("object");
            if (clsClasses_.empty())
                clsClasses_.push_back("class_0");
        }

        void initModels()
        {
            yolo_ = YoloNet((int)detClasses_.size(), detS_, detB_);
            cnn_ = CnnNet((int)clsClasses_.size(), 128);
            yolo_->to(torch::kCPU);
            cnn_->to(torch::kCPU);
            yoloReady_ = true;
            cnnReady_ = true;
        }

        torch::Tensor matToTensor(const cv::Mat &img, int size)
        {
            cv::Mat rgb;
            cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
            cv::Mat resized;
            cv::resize(rgb, resized, cv::Size(size, size));
            resized.convertTo(resized, CV_32F, 1.0 / 255.0);
            auto tensor = torch::from_blob(resized.data, {size, size, 3}, torch::kFloat32).clone();
            tensor = tensor.permute({2, 0, 1});
            return tensor;
        }

        torch::Tensor buildYoloTarget(const std::vector<DetBox> &boxes, int S, int B, int C)
        {
            auto target = torch::zeros({S, S, B * 5 + C}, torch::kFloat32);
            for (const auto &b : boxes)
            {
                int gx = std::min(S - 1, std::max(0, (int)(b.cx * S)));
                int gy = std::min(S - 1, std::max(0, (int)(b.cy * S)));
                float rx = b.cx * S - gx;
                float ry = b.cy * S - gy;
                int boxOffset = 0;
                target[gy][gx][boxOffset + 0] = rx;
                target[gy][gx][boxOffset + 1] = ry;
                target[gy][gx][boxOffset + 2] = b.w;
                target[gy][gx][boxOffset + 3] = b.h;
                target[gy][gx][boxOffset + 4] = 1.0f;
                int clsIdx = B * 5 + std::max(0, std::min(C - 1, b.cls));
                target[gy][gx][clsIdx] = 1.0f;
            }
            return target;
        }

        torch::Tensor yoloLossTensor(const torch::Tensor &pred, const torch::Tensor &target)
        {
            auto p = pred.permute({0, 2, 3, 1});
            auto t = target;
            int B = detB_;
            int classStart = B * 5;
            auto objMask = (t.slice(3, 4, 5) > 0.5f).to(torch::kFloat32);
            auto noMask = (t.slice(3, 4, 5) <= 0.5f).to(torch::kFloat32);
            auto pCoord = torch::sigmoid(p.slice(3, 0, 4));
            auto tCoord = t.slice(3, 0, 4);
            auto coordLoss = torch::mse_loss(pCoord, tCoord, torch::Reduction::None) * objMask;
            auto confPred = torch::sigmoid(p.slice(3, 4, 5));
            auto confLoss = torch::mse_loss(confPred, t.slice(3, 4, 5), torch::Reduction::None);
            auto objLoss = confLoss * objMask;
            auto noLoss = confLoss * noMask;
            auto clsLoss = torch::mse_loss(p.slice(3, classStart, p.size(3)), t.slice(3, classStart, t.size(3)), torch::Reduction::None);
            clsLoss = clsLoss * objMask;
            return coordLoss.mean() * 5.0f + objLoss.mean() * 1.0f + noLoss.mean() * 0.5f + clsLoss.mean() * 1.0f;
        }

        void trainAll()
        {
            trainYolo();
            trainCnn();
        }

        void trainYolo()
        {
            if (detItems_.empty())
                return;
            yolo_->train();
            torch::optim::Adam opt(yolo_->parameters(), torch::optim::AdamOptions(lr_));
            int total = (int)detItems_.size();
            for (int e = 0; e < detEpochs_; e++)
            {
                float epochLoss = 0.0f;
                for (int i = 0; i < total; i++)
                {
                    const auto &item = detItems_[i];
                    cv::Mat img = cv::imread(item.path.string(), cv::IMREAD_COLOR);
                    if (img.empty())
                        continue;
                    auto x = matToTensor(img, detInput_).unsqueeze(0);
                    auto t = buildYoloTarget(item.boxes, detS_, detB_, (int)detClasses_.size()).unsqueeze(0);
                    opt.zero_grad();
                    auto pred = yolo_->forward(x);
                    auto lossTensor = yoloLossTensor(pred, t);
                    lossTensor.backward();
                    opt.step();
                    epochLoss += lossTensor.item<float>();
                }
                (void)epochLoss;
            }
            yolo_->eval();
        }

        void trainCnn()
        {
            if (clsItems_.empty())
                return;
            cnn_->train();
            torch::optim::Adam opt(cnn_->parameters(), torch::optim::AdamOptions(lr_));
            int total = (int)clsItems_.size();
            for (int e = 0; e < clsEpochs_; e++)
            {
                float epochLoss = 0.0f;
                for (int i = 0; i < total; i++)
                {
                    const auto &item = clsItems_[i];
                    cv::Mat img = cv::imread(item.path.string(), cv::IMREAD_COLOR);
                    if (img.empty())
                        continue;
                    auto x = matToTensor(img, clsInput_).unsqueeze(0);
                    auto y = torch::tensor({item.label}, torch::kLong);
                    auto out = cnn_->forwardWithEmbedding(x).first;
                    auto loss = torch::nn::functional::cross_entropy(out, y);
                    opt.zero_grad();
                    loss.backward();
                    opt.step();
                    epochLoss += loss.item<float>();
                }
                (void)epochLoss;
            }
            cnn_->eval();
        }

        bool loadWeights()
        {
            bool ok = true;
            fs::path yPath = modelDir_ / "yolo_torch.pt";
            fs::path cPath = modelDir_ / "cnn_torch.pt";
            try
            {
                if (fs::exists(yPath))
                    torch::load(yolo_, yPath.string());
                else
                    ok = false;
            }
            catch (...)
            {
                ok = false;
            }
            try
            {
                if (fs::exists(cPath))
                    torch::load(cnn_, cPath.string());
                else
                    ok = false;
            }
            catch (...)
            {
                ok = false;
            }
            return ok;
        }

        void saveWeights()
        {
            fs::path yPath = modelDir_ / "yolo_torch.pt";
            fs::path cPath = modelDir_ / "cnn_torch.pt";
            try
            {
                torch::save(yolo_, yPath.string());
            }
            catch (...)
            {
            }
            try
            {
                torch::save(cnn_, cPath.string());
            }
            catch (...)
            {
            }
        }

        std::vector<DetBox> inferYolo(const cv::Mat &img)
        {
            torch::NoGradGuard guard;
            std::vector<DetBox> out;
            auto x = matToTensor(img, detInput_).unsqueeze(0);
            auto pred = yolo_->forward(x);
            auto p = pred.squeeze(0).permute({1, 2, 0});
            int S = detS_;
            int C = (int)detClasses_.size();
            for (int gy = 0; gy < S; gy++)
            {
                for (int gx = 0; gx < S; gx++)
                {
                    auto cell = p[gy][gx];
                    float conf = torch::sigmoid(cell[4]).item<float>();
                    if (conf < detScore_)
                        continue;
                    float rx = torch::sigmoid(cell[0]).item<float>();
                    float ry = torch::sigmoid(cell[1]).item<float>();
                    float w = torch::relu(cell[2]).item<float>();
                    float h = torch::relu(cell[3]).item<float>();
                    auto clsScores = cell.slice(0, detB_ * 5, detB_ * 5 + C);
                    int cls = clsScores.argmax().item<int>();
                    float cx = (gx + rx) / S * img.cols;
                    float cy = (gy + ry) / S * img.rows;
                    DetBox b;
                    b.cls = cls;
                    b.conf = conf;
                    b.cx = cx;
                    b.cy = cy;
                    b.w = w * img.cols;
                    b.h = h * img.rows;
                    out.push_back(b);
                }
            }
            return nms(out, detNms_);
        }

        static float iou(const DetBox &a, const DetBox &b)
        {
            float ax1 = a.cx - a.w * 0.5f;
            float ay1 = a.cy - a.h * 0.5f;
            float ax2 = a.cx + a.w * 0.5f;
            float ay2 = a.cy + a.h * 0.5f;
            float bx1 = b.cx - b.w * 0.5f;
            float by1 = b.cy - b.h * 0.5f;
            float bx2 = b.cx + b.w * 0.5f;
            float by2 = b.cy + b.h * 0.5f;
            float ix1 = std::max(ax1, bx1);
            float iy1 = std::max(ay1, by1);
            float ix2 = std::min(ax2, bx2);
            float iy2 = std::min(ay2, by2);
            float iw = std::max(0.0f, ix2 - ix1);
            float ih = std::max(0.0f, iy2 - iy1);
            float inter = iw * ih;
            float ua = a.w * a.h + b.w * b.h - inter;
            return ua <= 0.0f ? 0.0f : inter / ua;
        }

        // 对检测框执行非极大值抑制。
        // 调用方式：YOLO 解码后在输出前调用以去除重叠框。
        // 实现思路：按置信度排序并抑制与高分框 IoU 过高的候选。
        // 注意事项：阈值过低会漏检，过高会保留重复框。
        static std::vector<DetBox> nms(std::vector<DetBox> boxes, float thresh)
        {
            std::sort(boxes.begin(), boxes.end(), [](const DetBox &a, const DetBox &b)
                      { return a.conf > b.conf; });
            std::vector<DetBox> keep;
            std::vector<bool> suppressed(boxes.size(), false);
            for (size_t i = 0; i < boxes.size(); i++)
            {
                if (suppressed[i])
                    continue;
                keep.push_back(boxes[i]);
                for (size_t j = i + 1; j < boxes.size(); j++)
                {
                    if (suppressed[j])
                        continue;
                    if (iou(boxes[i], boxes[j]) > thresh)
                        suppressed[j] = true;
                }
            }
            return keep;
        }

        // 使用 CNN 分支提取细分类结果与图像嵌入。
        // 调用方式：视觉主流程 analyze 在检测后调用。
        // 实现思路：前向推理 logits、取 topk，再归一化 embedding。
        // 注意事项：要求 cnn_ 已初始化且输入图像非空。
        CnnDetail inferCnn(const cv::Mat &img)
        {
            torch::NoGradGuard guard;
            CnnDetail out;
            auto x = matToTensor(img, clsInput_).unsqueeze(0);
            auto pair = cnn_->forwardWithEmbedding(x);
            auto logits = pair.first.squeeze(0);
            auto emb = pair.second.squeeze(0);
            auto probs = torch::softmax(logits, 0);
            auto topk = std::get<1>(probs.topk(5));
            for (int i = 0; i < topk.size(0); i++)
            {
                int idx = topk[i].item<int>();
                float score = probs[idx].item<float>();
                out.topk.push_back({className(idx, clsClasses_), score});
            }
            out.embedding.resize((size_t)emb.size(0));
            std::memcpy(out.embedding.data(), emb.data_ptr<float>(), sizeof(float) * out.embedding.size());
            float norm = 0.0f;
            for (float v : out.embedding)
                norm += v * v;
            norm = std::sqrt(std::max(1e-9f, norm));
            for (auto &v : out.embedding)
                v /= norm;
            return out;
        }

        // 将类别索引映射为可读标签。
        // 调用方式：检测/分类结果序列化输出时调用。
        // 实现思路：优先读取标签表，越界时退化为 class_x。
        // 注意事项：labels 为空时仍可得到稳定可读名称。
        static std::string className(int idx, const std::vector<std::string> &names)
        {
            if (idx >= 0 && idx < (int)names.size())
                return names[idx];
            return "class_" + std::to_string(idx);
        }

        // 将视觉结果拼装为图上下文字符串。
        // 调用方式：返回给上游 NLP/图模块做联合推理提示。
        // 实现思路：按 labels/details 片段拼接成紧凑文本。
        // 注意事项：该文本是摘要格式，不包含完整像素信息。
        std::string buildGraphContext(const std::vector<DetBox> &dets,
                                      const std::vector<std::pair<std::string, float>> &details) const
        {
            std::ostringstream oss;
            oss << "vision";
            if (!dets.empty())
            {
                oss << "|labels:";
                for (size_t i = 0; i < dets.size(); i++)
                {
                    if (i)
                        oss << ",";
                    oss << className(dets[i].cls, detClasses_) << "(" << std::fixed << std::setprecision(2) << dets[i].conf << ")";
                }
            }
            if (!details.empty())
            {
                oss << "|details:";
                for (size_t i = 0; i < details.size(); i++)
                {
                    if (i)
                        oss << ",";
                    oss << details[i].first << "(" << std::fixed << std::setprecision(2) << details[i].second << ")";
                }
            }
            return oss.str();
        }

        fs::path dataRoot_;
        fs::path modelDir_;
        int clsInput_{224};
        int detInput_{416};
        int detS_{7};
        int detB_{2};
        float detScore_{0.25f};
        float detNms_{0.45f};
        int detEpochs_{10};
        int clsEpochs_{8};
        int batch_{8};
        float lr_{0.001f};
        bool forceTrain_{false};

        std::vector<ClsItem> clsItems_;
        std::vector<DetItem> detItems_;
        std::vector<std::string> clsClasses_;
        std::vector<std::string> detClasses_;

        YoloNet yolo_{nullptr};
        CnnNet cnn_{nullptr};
        bool yoloReady_{false};
        bool cnnReady_{false};
        bool trained_{false};
        std::mutex trainMu_;
    };
#endif

    // 将 Base64 文本解码到输出字节缓冲区。
    // 调用方式：语音与图像接口解码请求体时调用。
    // 实现思路：按字符查表累积 6bit 并回写 8bit 字节。
    // 注意事项：非法字符会被跳过，调用方需自行校验业务完整性。
    static void base64DecodeTo(const std::string &input, std::vector<uint8_t> &out)
    {
        static const int kDecTable[256] = {
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -2, -1, -1,
            -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1,
            -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        out.clear();
        out.reserve(input.size() * 3 / 4);
        int val = 0;
        int valb = -8;
        for (unsigned char c : input)
        {
            if (std::isspace(c))
                continue;
            int d = kDecTable[c];
            if (d == -1)
                continue;
            if (d == -2)
                break;
            val = (val << 6) + d;
            valb += 6;
            if (valb >= 0)
            {
                out.push_back((uint8_t)((val >> valb) & 0xFF));
                valb -= 8;
            }
        }
    }

    // 返回式 Base64 解码封装。
    // 调用方式：当调用方希望直接拿到新分配的字节数组时使用。
    // 实现思路：内部复用 base64DecodeTo 填充并返回结果。
    // 注意事项：对大输入会产生额外拷贝和分配成本。
    static std::vector<uint8_t> base64Decode(const std::string &input)
    {
        std::vector<uint8_t> out;
        base64DecodeTo(input, out);
        return out;
    }

    // VisionPipeline 封装基于 OpenCV DNN 的视觉推理流程。
    // 调用方式：通过 inferImage 输入图像并返回检测/分类结果。
    // 实现思路：维护 YOLO/CNN 模型、预处理和后处理配置。
    // 注意事项：模型路径或标签缺失会导致对应分支不可用。
    // 注意事项：并发推理依赖内部互斥锁保护底层网络对象。
    // 注意事项：该管线与 Torch 版本可并存，由配置决定使用。
    struct VisionPipeline
    {
        // Detection 表示单个目标检测输出项。
        // 调用方式：YOLO 后处理阶段生成并汇总到结果数组。
        // 实现思路：包含标签、置信度与像素坐标框。
        // 注意事项：box 基于当前图像尺寸，跨尺寸需重映射。
        // 注意事项：score 阈值过滤在上游后处理中完成。
        // 注意事项：该结构为值类型，便于排序与序列化。
        struct Detection
        {
            std::string label;
            float score{0.0f};
            cv::Rect box;
        };

        // 构造 OpenCV 视觉管线并加载配置。
        // 调用方式：服务启动时创建单例对象。
        // 实现思路：读取环境变量、准备标签和可选 Torch 分支。
        // 注意事项：当前实现允许无外部模型的启发式兜底分析。
        VisionPipeline()
        {
            yoloModel_ = resolveConfig<std::string>("vision.yolo.model", std::string(""), "VISION_YOLO_MODEL");
            yoloLabels_ = resolveConfig<std::string>("vision.yolo.labels", std::string(""), "VISION_YOLO_LABELS");
            yoloInput_ = std::max(128, resolveConfig<int>("vision.yolo.input", 640, "VISION_YOLO_INPUT"));
            // Runtime-tunable thresholds from config/phoenix.json (env may override)
            try { yoloScore_ = resolveConfig<float>("vision.confidenceThreshold", 0.35f); } catch (...) { yoloScore_ = 0.35f; }
            try { yoloNms_ = resolveConfig<float>("vision.nmsThreshold", 0.45f); } catch (...) { yoloNms_ = 0.45f; }
            try { yoloMax_ = resolveConfig<int>("vision.maxBoxes", 80); } catch (...) { yoloMax_ = 80; }
            try { minBoxAreaRatio_ = resolveConfig<float>("vision.minBoxAreaRatio", 0.001f); } catch (...) { minBoxAreaRatio_ = 0.001f; }

            cnnModel_ = resolveConfig<std::string>("vision.cnn.model", std::string(""), "VISION_CNN_MODEL");
            cnnLabels_ = resolveConfig<std::string>("vision.cnn.labels", std::string(""), "VISION_CNN_LABELS");
            cnnInput_ = std::max(64, resolveConfig<int>("vision.cnn.input", 224, "VISION_CNN_INPUT"));
            cnnTopK_ = std::max(1, resolveConfig<int>("vision.cnn.topK", 5, "VISION_CNN_TOPK"));
            cnnEmbedLayer_ = resolveConfig<std::string>("vision.cnn.embedLayer", std::string(""), "VISION_CNN_EMBED_LAYER");
            cnnMean_ = resolveConfig<std::vector<float>>("vision.cnn.mean", std::vector<float>{0.0f, 0.0f, 0.0f}, "VISION_CNN_MEAN");
            cnnStd_ = resolveConfig<std::vector<float>>("vision.cnn.std", std::vector<float>{1.0f, 1.0f, 1.0f}, "VISION_CNN_STD");
            if (cnnMean_.size() != 3)
                cnnMean_ = {0.0f, 0.0f, 0.0f};
            if (cnnStd_.size() != 3)
                cnnStd_ = {1.0f, 1.0f, 1.0f};

            // 禁止依赖外部模型文件，使用内置的确定性视觉分析流程
            yoloReady_ = true;
            cnnReady_ = true;
            if (!yoloLabels_.empty())
                yoloClassNames_ = loadLabels(yoloLabels_);
            if (!cnnLabels_.empty())
                cnnClassNames_ = loadLabels(cnnLabels_);
#ifdef HAVE_TORCH
            torchPipeline_ = std::make_shared<TorchVisionPipeline>();
#endif
        }

        // 判断视觉管线是否具备可执行能力。
        // 调用方式：请求进入推理路径前调用。
        // 实现思路：综合 OpenCV 分支与可选 Torch 分支状态。
        // 注意事项：ready 只表示可运行，不代表结果质量。
        bool ready() const
        {
            bool baseReady = yoloReady_ || cnnReady_;
#ifdef HAVE_TORCH
            if (torchPipeline_)
            {
                return torchPipeline_->ready() || baseReady;
            }
#endif
            return baseReady;
        }

        // 执行图像分析并输出统一 JSON 结果。
        // 调用方式：/vision/analyze 等接口直接调用。
        // 实现思路：优先 Torch，失败回退到 OpenCV 启发式流程。
        // 注意事项：空图像会返回错误且不进入推理。
        Json::Value analyze(const cv::Mat &img)
        {
#ifdef HAVE_TORCH
            if (torchPipeline_)
            {
                auto res = torchPipeline_->analyze(img);
                if (res.isMember("ok") && res["ok"].asBool())
                    return res;
            }
#endif
            Json::Value out;
            out["ok"] = false;
            if (img.empty())
            {
                out["error"] = "empty image";
                return out;
            }
            std::vector<Detection> dets;
            std::vector<std::pair<std::string, float>> details;
            std::vector<float> embedding;
            if (yoloReady_)
            {
                dets = runYolo(img);
            }
            if (cnnReady_)
            {
                details = runCnn(img, embedding);
            }

            Json::Value detArr(Json::arrayValue);
            for (const auto &d : dets)
            {
                Json::Value item;
                item["label"] = d.label;
                item["score"] = d.score;
                Json::Value box(Json::arrayValue);
                box.append(d.box.x);
                box.append(d.box.y);
                box.append(d.box.width);
                box.append(d.box.height);
                item["box"] = box;
                detArr.append(item);
            }

            Json::Value detDetail(Json::arrayValue);
            for (const auto &p : details)
            {
                Json::Value item;
                item["label"] = p.first;
                item["score"] = p.second;
                detDetail.append(item);
            }

            Json::Value emb(Json::arrayValue);
            for (float v : embedding)
                emb.append(v);

            out["ok"] = true;
            out["detections"] = detArr;
            out["details"] = detDetail;
            out["embedding"] = emb;
            out["embeddingDim"] = (int)embedding.size();
            out["imageSize"] = Json::Value(Json::objectValue);
            out["imageSize"]["w"] = img.cols;
            out["imageSize"]["h"] = img.rows;
            out["graphContext"] = buildGraphContext(dets, details);
            return out;
        }

    private:
        // 从标签文件加载类别名称列表。
        // 调用方式：构造时读取 yolo/cnn 标签配置。
        // 实现思路：逐行读取非空文本并按顺序保存。
        // 注意事项：文件不可读时返回空列表。
        static std::vector<std::string> loadLabels(const std::string &path)
        {
            std::vector<std::string> labels;
            std::ifstream in(path);
            if (!in)
                return labels;
            std::string line;
            while (std::getline(in, line))
            {
                if (!line.empty())
                    labels.push_back(line);
            }
            return labels;
        }

        // 将逗号分隔字符串解析为浮点数组。
        // 调用方式：读取均值/方差等配置项时调用。
        // 实现思路：按 ',' 切分并逐项 stof，失败填 0。
        // 注意事项：输入格式异常会降级，不抛异常。
        static std::vector<float> parseFloatList(const std::string &text)
        {
            std::vector<float> out;
            std::stringstream ss(text);
            std::string tok;
            while (std::getline(ss, tok, ','))
            {
                try
                {
                    out.push_back(std::stof(tok));
                }
                catch (...)
                {
                    out.push_back(0.0f);
                }
            }
            return out;
        }

        // 运行启发式 YOLO 替代检测流程。
        // 调用方式：当 yoloReady_ 为 true 时由 analyze 调用。
        // 实现思路：边缘/轮廓提取后按面积与形状生成检测框。
        // 注意事项：这是近似检测逻辑，适合无模型兜底场景。
        std::vector<Detection> runYolo(const cv::Mat &img)
        {
            std::vector<Detection> dets;
            if (!yoloReady_)
                return dets;
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(yoloInput_, yoloInput_));
            cv::Mat gray;
            cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.2);
            cv::Mat edges;
            cv::Canny(gray, edges, 60, 180);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            std::vector<Detection> candidates;
            const float imgArea = static_cast<float>(resized.cols * resized.rows);
            for (const auto &c : contours)
            {
                cv::Rect r = cv::boundingRect(c);
                float area = static_cast<float>(r.area());
                if (area < imgArea * minBoxAreaRatio_)
                    continue;
                if (r.width < 6 || r.height < 6)
                    continue;
                float score = std::min(1.0f, area / imgArea * 4.0f);
                if (score < yoloScore_)
                    continue;
                Detection d;
                d.box = cv::Rect(
                    r.x * img.cols / resized.cols,
                    r.y * img.rows / resized.rows,
                    r.width * img.cols / resized.cols,
                    r.height * img.rows / resized.rows);
                float ar = r.width > 0 ? (float)r.height / (float)r.width : 1.0f;
                if (ar > 1.6f)
                    d.label = "tall_object";
                else if (ar < 0.6f)
                    d.label = "wide_object";
                else
                    d.label = "object";
                d.score = score;
                candidates.push_back(std::move(d));
            }

            std::sort(candidates.begin(), candidates.end(), [](const Detection &a, const Detection &b)
                      { return a.score > b.score; });
            int keep = std::min((int)candidates.size(), yoloMax_);
            dets.assign(candidates.begin(), candidates.begin() + keep);
            return dets;
        }

        // 运行启发式 CNN 细节分类与嵌入构建。
        // 调用方式：由 analyze 在 cnnReady_ 条件下调用。
        // 实现思路：基于颜色/纹理统计生成标签分数与向量。
        // 注意事项：输出是规则特征，不等价真实 CNN 模型结果。
        std::vector<std::pair<std::string, float>> runCnn(const cv::Mat &img, std::vector<float> &embedding)
        {
            std::vector<std::pair<std::string, float>> out;
            if (!cnnReady_)
                return out;
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(cnnInput_, cnnInput_));
            cv::Mat hsv;
            cv::cvtColor(resized, hsv, cv::COLOR_BGR2HSV);
            cv::Scalar meanBgr, stdBgr;
            cv::meanStdDev(resized, meanBgr, stdBgr);

            cv::Scalar meanHsv = cv::mean(hsv);
            float brightness = (float)meanHsv[2] / 255.0f;
            float saturation = (float)meanHsv[1] / 255.0f;
            float warm = std::max(0.0f, (float)(meanBgr[2] - meanBgr[0]) / 255.0f);
            float cool = std::max(0.0f, (float)(meanBgr[0] - meanBgr[2]) / 255.0f);

            cv::Mat gray;
            cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
            cv::Mat lap;
            cv::Laplacian(gray, lap, CV_32F);
            cv::Scalar lapMean, lapStd;
            cv::meanStdDev(lap, lapMean, lapStd);
            float texture = std::min(1.0f, (float)(lapStd[0] / 64.0));

            // 细化标签（启发式评分）
            std::vector<std::pair<std::string, float>> scores;
            scores.push_back({"bright_scene", brightness});
            scores.push_back({"dark_scene", 1.0f - brightness});
            scores.push_back({"high_saturation", saturation});
            scores.push_back({"low_saturation", 1.0f - saturation});
            scores.push_back({"warm_tone", warm});
            scores.push_back({"cool_tone", cool});
            scores.push_back({"high_texture", texture});
            scores.push_back({"smooth_texture", 1.0f - texture});

            std::sort(scores.begin(), scores.end(), [](auto &a, auto &b)
                      { return a.second > b.second; });
            int keep = std::min((int)scores.size(), cnnTopK_);
            out.assign(scores.begin(), scores.begin() + keep);

            // 构建稳定嵌入
            embedding.clear();
            embedding.reserve(3 + 3 + 8 + 4);
            embedding.push_back((float)meanBgr[2] / 255.0f);
            embedding.push_back((float)meanBgr[1] / 255.0f);
            embedding.push_back((float)meanBgr[0] / 255.0f);
            embedding.push_back((float)stdBgr[2] / 255.0f);
            embedding.push_back((float)stdBgr[1] / 255.0f);
            embedding.push_back((float)stdBgr[0] / 255.0f);
            embedding.push_back(brightness);
            embedding.push_back(saturation);
            embedding.push_back(warm);
            embedding.push_back(cool);
            embedding.push_back(texture);

            float norm = 0.0f;
            for (float v : embedding)
                norm += v * v;
            norm = std::sqrt(std::max(1e-9f, norm));
            for (auto &v : embedding)
                v /= norm;
            return out;
        }

        // 将检测与细节结果转换为图上下文文本。
        // 调用方式：视觉结果回传到图推理或聊天上下文时调用。
        // 实现思路：拼接 labels/details 片段形成紧凑描述串。
        // 注意事项：输出面向机器消费，非终端用户文案。
        static std::string buildGraphContext(const std::vector<Detection> &dets,
                                             const std::vector<std::pair<std::string, float>> &details)
        {
            std::ostringstream oss;
            oss << "vision";
            if (!dets.empty())
            {
                oss << "|labels:";
                for (size_t i = 0; i < dets.size(); i++)
                {
                    if (i)
                        oss << ",";
                    oss << dets[i].label << "(" << std::fixed << std::setprecision(2) << dets[i].score << ")";
                }
            }
            if (!details.empty())
            {
                oss << "|details:";
                for (size_t i = 0; i < details.size(); i++)
                {
                    if (i)
                        oss << ",";
                    oss << details[i].first << "(" << std::fixed << std::setprecision(2) << details[i].second << ")";
                }
            }
            return oss.str();
        }

        cv::dnn::Net yoloNet_;
        cv::dnn::Net cnnNet_;
        std::mutex yoloMu_;
        std::mutex cnnMu_;
        bool yoloReady_{false};
        bool cnnReady_{false};
        std::string yoloModel_;
        std::string yoloLabels_;
        int yoloInput_{640};
        float yoloScore_{0.35f};
        float yoloNms_{0.45f};
        int yoloMax_{80};
        float minBoxAreaRatio_{0.001f};
        std::string cnnModel_;
        std::string cnnLabels_;
        int cnnInput_{224};
        int cnnTopK_{5};
        std::string cnnEmbedLayer_;
        std::vector<std::string> yoloClassNames_;
        std::vector<std::string> cnnClassNames_;
        std::vector<float> cnnMean_{0.0f, 0.0f, 0.0f};
        std::vector<float> cnnStd_{1.0f, 1.0f, 1.0f};
    };

#ifdef HAVE_TORCH
    // TorchVocab 维护文本模型词表与编码规则。
    // 调用方式：先 build，再用 encode 将 token 序列转 id。
    // 实现思路：按词频构建 stoi/itos 并保留 pad/unk 特殊词。
    // 注意事项：maxVocab 控制词表上限，超出词降为 unk。
    // 注意事项：编码长度由调用方提供 maxLen 截断或补齐。
    // 注意事项：词表变更后旧模型需重新训练或对齐。
    struct TorchVocab
    {
        int64_t padId{0};
        int64_t unkId{1};
        int64_t maxVocab{8000};
        std::unordered_map<std::string, int64_t> stoi;
        std::vector<std::string> itos;

        void build(const std::unordered_map<std::string, int> &freq)
        {
            std::vector<std::pair<std::string, int>> items(freq.begin(), freq.end());
            std::sort(items.begin(), items.end(), [](auto &a, auto &b)
                      { return a.second > b.second; });
            itos.clear();
            stoi.clear();
            itos.push_back("<pad>");
            itos.push_back("<unk>");
            for (size_t i = 0; i < items.size() && (int64_t)itos.size() < maxVocab; i++)
            {
                stoi[items[i].first] = (int64_t)itos.size();
                itos.push_back(items[i].first);
            }
        }

        std::vector<int64_t> encode(const std::vector<std::string> &tokens, int64_t maxLen) const
        {
            std::vector<int64_t> ids(maxLen, padId);
            int64_t idx = 0;
            for (const auto &t : tokens)
            {
                if (idx >= maxLen)
                    break;
                auto it = stoi.find(t);
                ids[idx++] = (it == stoi.end()) ? unkId : it->second;
            }
            return ids;
        }
    };

    // TorchGRUImpl 定义基于 GRU 的文本序列建模网络。
    // 调用方式：训练调用 forward，构建语义向量调用 hidden。
    // 实现思路：Embedding -> GRU -> Linear 输出逐位置 logits。
    // 注意事项：输入 id 必须在词表范围内。
    // 注意事项：batch_first=true，输入形状应为 [B,T]。
    // 注意事项：hidden 返回最后时间步表示。
    struct TorchGRUImpl : torch::nn::Module
    {
        torch::nn::Embedding emb{nullptr};
        torch::nn::GRU gru{nullptr};
        torch::nn::Linear fc{nullptr};

        TorchGRUImpl(int64_t vocab, int64_t embDim, int64_t hidDim)
            : emb(register_module("emb", torch::nn::Embedding(vocab, embDim))),
              gru(register_module("gru", torch::nn::GRU(torch::nn::GRUOptions(embDim, hidDim).batch_first(true)))),
              fc(register_module("fc", torch::nn::Linear(hidDim, vocab))) {}

        torch::Tensor forward(torch::Tensor x)
        {
            auto e = emb(x);
            auto out = std::get<0>(gru(e));
            auto logits = fc(out);
            return logits;
        }

        torch::Tensor hidden(torch::Tensor x)
        {
            auto e = emb(x);
            auto out = std::get<0>(gru(e));
            return out.select(1, out.size(1) - 1);
        }
    };
    TORCH_MODULE(TorchGRU);

    // TorchLSTMImpl 定义基于 LSTM 的文本序列建模网络。
    // 调用方式：可用于训练语言模型或提取句向量。
    // 实现思路：Embedding -> LSTM -> Linear 生成词表分布。
    // 注意事项：输入输出维度需与 TorchVocab 和配置一致。
    // 注意事项：hidden 同样取最后时间步作为表示。
    // 注意事项：该模块不包含采样策略，由上层控制。
    struct TorchLSTMImpl : torch::nn::Module
    {
        torch::nn::Embedding emb{nullptr};
        torch::nn::LSTM lstm{nullptr};
        torch::nn::Linear fc{nullptr};

        TorchLSTMImpl(int64_t vocab, int64_t embDim, int64_t hidDim)
            : emb(register_module("emb", torch::nn::Embedding(vocab, embDim))),
              lstm(register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(embDim, hidDim).batch_first(true)))),
              fc(register_module("fc", torch::nn::Linear(hidDim, vocab))) {}

        torch::Tensor forward(torch::Tensor x)
        {
            auto e = emb(x);
            auto out = std::get<0>(lstm(e));
            auto logits = fc(out);
            return logits;
        }

        torch::Tensor hidden(torch::Tensor x)
        {
            auto e = emb(x);
            auto out = std::get<0>(lstm(e));
            return out.select(1, out.size(1) - 1);
        }
    };
    TORCH_MODULE(TorchLSTM);

    // TorchTextModels 聚合 Torch 文本模型及训练配置。
    // 调用方式：初始化后可训练并对文本生成 embedding 表示。
    // 实现思路：维护词表、GRU/LSTM 模型与训练超参数。
    // 注意事项：ready=false 时调用方应走非 Torch 回退路径。
    // 注意事项：训练语料规模和质量直接影响嵌入效果。
    // 注意事项：该结构包含模型状态，避免频繁复制。
    struct TorchTextModels
    {
        bool ready{false};
        TorchVocab vocab;
        int64_t maxLen{64};
        int64_t embDim{128};
        int64_t hidDim{128};
        int epochs{4};
        int batch{16};
        TorchGRU gru{nullptr};
        TorchLSTM lstm{nullptr};

        void initFromCorpus(const fs::path &robotsDir, int maxFiles)
        {
            std::unordered_map<std::string, int> freq;
            std::vector<std::vector<std::string>> samples;
            if (fs::exists(robotsDir))
            {
                int fileCount = 0;
                for (const auto &entry : fs::directory_iterator(robotsDir))
                {
                    if (!entry.is_regular_file())
                        continue;
                    if (entry.path().extension() != ".txt")
                        continue;
                    std::ifstream in(entry.path(), std::ios::binary);
                    if (!in)
                        continue;
                    std::ostringstream ss;
                    ss << in.rdbuf();
                    auto tokens = Tokenizer::tokenize(ss.str());
                    if (!tokens.empty())
                        samples.push_back(tokens);
                    for (const auto &t : tokens)
                        freq[t] += 1;
                    if (++fileCount >= maxFiles)
                        break;
                }
            }
            vocab.build(freq);
            int64_t vocabSize = std::max<int64_t>(2, (int64_t)vocab.itos.size());
            gru = TorchGRU(vocabSize, embDim, hidDim);
            lstm = TorchLSTM(vocabSize, embDim, hidDim);
            trainOnSamples(samples, vocabSize);
        }

        std::vector<float> embedText(const std::vector<std::string> &tokens)
        {
            std::vector<float> out((size_t)embDim, 0.0f);
            if (!ready || tokens.empty())
                return out;
            auto ids = vocab.encode(tokens, std::min<int64_t>(maxLen, (int64_t)tokens.size()));
            torch::NoGradGuard guard;
            auto x = torch::from_blob(ids.data(), {(int64_t)1, (int64_t)ids.size()}, torch::kInt64).clone();
            auto emb = gru->emb->forward(x).mean(1).squeeze(0).contiguous();
            out.resize((size_t)emb.size(0));
            std::memcpy(out.data(), emb.data_ptr<float>(), sizeof(float) * out.size());
            normalize(out);
            return out;
        }

        torch::Tensor hiddenGru(const std::vector<std::string> &tokens)
        {
            auto ids = vocab.encode(tokens, maxLen);
            auto x = torch::from_blob(ids.data(), {1, (int64_t)ids.size()}, torch::kInt64).clone();
            return gru->hidden(x);
        }

        torch::Tensor hiddenLstm(const std::vector<std::string> &tokens)
        {
            auto ids = vocab.encode(tokens, maxLen);
            auto x = torch::from_blob(ids.data(), {1, (int64_t)ids.size()}, torch::kInt64).clone();
            return lstm->hidden(x);
        }

    private:
        void trainOnSamples(const std::vector<std::vector<std::string>> &samples, int64_t vocabSize)
        {
            if (samples.empty())
            {
                ready = true;
                return;
            }
            torch::optim::Adam optGru(gru->parameters(), torch::optim::AdamOptions(1e-3));
            torch::optim::Adam optLstm(lstm->parameters(), torch::optim::AdamOptions(1e-3));
            std::vector<std::vector<int64_t>> seqs;
            for (const auto &s : samples)
            {
                if (s.size() < 2)
                    continue;
                auto ids = vocab.encode(s, (int64_t)std::max<int64_t>(maxLen, 4));
                if (ids.size() < 2)
                    continue;
                while (ids.size() < (size_t)maxLen)
                    ids.push_back(vocab.padId);
                seqs.push_back(std::move(ids));
            }
            if (seqs.empty())
            {
                ready = true;
                return;
            }
            for (int e = 0; e < epochs; e++)
            {
                for (size_t i = 0; i < seqs.size(); i += (size_t)batch)
                {
                    size_t end = std::min(seqs.size(), i + (size_t)batch);
                    int64_t bsz = (int64_t)(end - i);
                    auto x = torch::zeros({bsz, maxLen}, torch::kInt64);
                    for (size_t b = 0; b < end - i; b++)
                    {
                        for (int64_t j = 0; j < maxLen; j++)
                        {
                            x.index_put_({(int64_t)b, j}, seqs[i + b][(size_t)j]);
                        }
                    }
                    auto input = x.narrow(1, 0, maxLen - 1);
                    auto target = x.narrow(1, 1, maxLen - 1).reshape({-1});

                    optGru.zero_grad();
                    auto logitsG = gru->forward(input).reshape({-1, vocabSize});
                    auto lossG = torch::nn::functional::cross_entropy(logitsG, target,
                                                                      torch::nn::functional::CrossEntropyFuncOptions().ignore_index(vocab.padId));
                    lossG.backward();
                    optGru.step();

                    optLstm.zero_grad();
                    auto logitsL = lstm->forward(input).reshape({-1, vocabSize});
                    auto lossL = torch::nn::functional::cross_entropy(logitsL, target,
                                                                      torch::nn::functional::CrossEntropyFuncOptions().ignore_index(vocab.padId));
                    lossL.backward();
                    optLstm.step();
                }
            }
            ready = true;
        }

        static void normalize(std::vector<float> &v)
        {
            double norm = 0.0;
            for (float x : v)
                norm += x * x;
            norm = std::sqrt(std::max(1e-9, norm));
            for (auto &x : v)
                x = (float)(x / norm);
        }
    };
#endif

    // ContextService 负责会话上下文管理与多模型提示生成。
    // 调用方式：服务层通过 ingest 接口输入文本并获取上下文。
    // 实现思路：维护 session 状态并融合 embedding/序列模型特征。
    // 注意事项：会话数据保存在内存，重启后不会自动恢复。
    // 注意事项：并发访问通过内部互斥锁序列化关键路径。
    // 注意事项：Torch 开关由构造参数与编译宏共同决定。
    class ContextService
    {
    public:
        // 异步流水线系统：各层独立处理不阻塞
        struct PipelineStage
        {
            std::vector<std::thread> workers;
            std::queue<std::function<void()>> taskQueue;
            std::mutex queueMutex;
            std::condition_variable queueCV;
            std::atomic<bool> shutdown{false};
            int workerCount{2};
        };

        explicit ContextService(const fs::path &robotsDir, bool useTorch)
            : embeddings_(), useTorchModels_(useTorch), robotsDir_(robotsDir)
        {
            // 跨 session 学习配置：探测基座 LLM 判定"已知/未知"，仅持久化未知事实。
            probeBaseUrl_ = resolveConfig<std::string>("knowledge_probe.llamacppBaseUrl", std::string("http://127.0.0.1:8082"), "FRONTEND_LLAMACPP_BASE_URL", "AI_LLAMACPP_BASE_URL");
            while (!probeBaseUrl_.empty() && probeBaseUrl_.back() == '/')
                probeBaseUrl_.pop_back();
            probeModel_ = resolveConfig<std::string>("knowledge_probe.llamacppModel", std::string(""), "FRONTEND_LLAMACPP_MODEL", "AI_LLAMACPP_MODEL");
            probeTimeoutMs_ = std::max(2000, resolveConfig<int>("knowledge_probe.probeTimeoutMs", 60000, "FRONTEND_KNOWLEDGE_PROBE_TIMEOUT_MS"));
            knownSimThreshold_ = resolveConfig<float>("knowledge_probe.knownSimThreshold", 0.5f, "FRONTEND_KNOWLEDGE_KNOWN_SIM");
            crossSessionLearnEnabled_ = resolveConfig<bool>("knowledge_probe.crossSessionLearnEnabled", false, "FRONTEND_CROSS_SESSION_LEARN");

            embeddings_.dim = std::max(32, resolveConfig<int>("context.embeddings.dim", 128, "FRONTEND_EMB_DIM"));
            embeddings_.window = std::max(1, resolveConfig<int>("context.embeddings.window", 4, "FRONTEND_EMB_WINDOW"));
            embeddings_.minCount = std::max(1, resolveConfig<int>("context.embeddings.minCount", 2, "FRONTEND_EMB_MIN_COUNT"));
            embeddings_.maxVocab = std::max(200, resolveConfig<int>("context.embeddings.maxVocab", 3000, "FRONTEND_EMB_MAX_VOCAB"));
            embeddings_.setCorpus(robotsDir, (size_t)std::max(10, resolveConfig<int>("context.embeddings.maxFiles", 400, "FRONTEND_EMB_MAX_FILES")));
#ifdef HAVE_TORCH
            if (useTorchModels_)
            {
                torchModels_.maxLen = std::max<int64_t>(16, resolveConfig<int64_t>("context.torch.maxLen", 64, "FRONTEND_TORCH_MAX_LEN"));
                torchModels_.embDim = std::max<int64_t>(32, resolveConfig<int64_t>("context.torch.embDim", 128, "FRONTEND_TORCH_EMB_DIM"));
                torchModels_.hidDim = std::max<int64_t>(32, resolveConfig<int64_t>("context.torch.hidDim", 128, "FRONTEND_TORCH_HID_DIM"));
                torchModels_.epochs = std::max(1, resolveConfig<int>("context.torch.epochs", 4, "FRONTEND_TORCH_EPOCHS"));
                torchModels_.batch = std::max(4, resolveConfig<int>("context.torch.batch", 16, "FRONTEND_TORCH_BATCH"));
                maxTorchFiles_ = std::max(10, resolveConfig<int>("context.torch.maxFiles", 240, "FRONTEND_TORCH_MAX_FILES"));
            }
#endif

            // 初始化异步流水线各阶段
            auto initStage = [this](PipelineStage &stage, const std::string &dotPath, const char *envVar, int defaultWorkers) {
                try { stage.workerCount = std::max(1, std::min(8, resolveConfig<int>(dotPath, defaultWorkers, envVar))); } catch (...) { stage.workerCount = defaultWorkers; }
                stage.shutdown.store(false);
                for (int i = 0; i < stage.workerCount; ++i)
                {
                    stage.workers.emplace_back([&stage]() {
                        while (true)
                        {
                            std::function<void()> task;
                            {
                                std::unique_lock<std::mutex> lock(stage.queueMutex);
                                stage.queueCV.wait(lock, [&stage]() { return stage.shutdown.load() || !stage.taskQueue.empty(); });
                                if (stage.shutdown.load() && stage.taskQueue.empty())
                                    return;
                                if (!stage.taskQueue.empty())
                                {
                                    task = std::move(stage.taskQueue.front());
                                    stage.taskQueue.pop();
                                }
                            }
                            if (task)
                                task();
                        }
                    });
                }
            };

            initStage(embeddingStage_, "frontend_server.embeddingWorkers", "FRONTEND_EMBEDDING_WORKERS", 2);
            initStage(rnnStage_, "frontend_server.rnnWorkers", "FRONTEND_RNN_WORKERS", 2);
            initStage(contextStage_, "frontend_server.contextWorkers", "FRONTEND_CONTEXT_WORKERS", 2);
            initStage(episodicStage_, "frontend_server.episodicWorkers", "FRONTEND_EPISODIC_WORKERS", 1);

            // 异步启动摘要模型：先尝试加载 checkpoint，否则从 WikiText 预训练。
            // FRONTEND_SUMMARY_MODEL_ENABLED=false 可完全禁用（默认启用）。
            {
                bool summaryEnabled = resolveConfig<bool>("summary_model.enabled", true, "FRONTEND_SUMMARY_MODEL_ENABLED");
                if (summaryEnabled)
                {
                    int pretrainLines = std::max(1000, resolveConfig<int>("summary_model.pretrainLines", 200000, "FRONTEND_SUMMARY_PRETRAIN_LINES"));
                    int trainEpochs = std::max(1, resolveConfig<int>("summary_model.trainEpochs", 4, "FRONTEND_SUMMARY_TRAIN_EPOCHS"));
                    fs::path wikitextPath = robotsDir_ / ".." / "robots" / "wikitext-103-all.txt";
                    if (!fs::exists(wikitextPath))
                        wikitextPath = robotsDir_ / "wikitext-103-all.txt";
                    submitToStage(episodicStage_, [this, wikitextPath, pretrainLines, trainEpochs]() {
                        if (!summaryModel_.tryLoadCheckpoint())
                        {
                            summaryModel_.trainAsync(wikitextPath, {}, pretrainLines, trainEpochs);
                        }
                    });
                }
            }
        }

        ~ContextService()
        {
            auto shutdownStage = [](PipelineStage &stage) {
                stage.shutdown.store(true);
                stage.queueCV.notify_all();
                for (auto &t : stage.workers)
                {
                    if (t.joinable())
                        t.join();
                }
            };
            shutdownStage(embeddingStage_);
            shutdownStage(rnnStage_);
            shutdownStage(contextStage_);
            shutdownStage(episodicStage_);
        }

        // 提交任务到指定流水线阶段
        void submitToStage(PipelineStage &stage, std::function<void()> task)
        {
            {
                std::lock_guard<std::mutex> lock(stage.queueMutex);
                stage.taskQueue.push(std::move(task));
            }
            stage.queueCV.notify_one();
        }

        Json::Value ingest(const std::string &sessionId, const std::string &text, const std::string &modeHint)
        {
            // Torch model warm-up can be expensive; do it before taking the session mutex
            // so one cold-start does not block all context sessions.
            ensureTorchReady();
            std::lock_guard<std::mutex> lock(mu_);
            auto &state = sessions_[sessionId];
            state.messageCount += 1;

            const auto tokens = Tokenizer::tokenize(text);
            const int tokenCount = static_cast<int>(tokens.size());
            std::string mode = selectMode(modeHint, tokenCount, state);
            state.shortWindow.pushUser(text);
            state.concat.push(text);
            const std::string shortContext = state.shortWindow.render();
            const std::string lexicalContext = state.concat.concat();

            // 异步流水线：各层独立处理不阻塞
            // 阶段1: embedding计算（异步）
            std::promise<std::vector<float>> embPromise;
            std::future<std::vector<float>> embFuture = embPromise.get_future();
            submitToStage(embeddingStage_, [this, text, tokens, &embPromise]() {
                auto emb = embeddings_.embedText(text);
#ifdef HAVE_TORCH
                if (useTorchModels_ && torchModels_.ready)
                {
                    emb = torchModels_.embedText(tokens);
                }
#endif
                embPromise.set_value(emb);
            });

            // 阶段2: RNN/LSTM上下文计算（异步，依赖embedding）
            std::promise<std::string> rnnPromise;
            std::future<std::string> rnnFuture = rnnPromise.get_future();
            std::string sessionIdCopy = sessionId;
            submitToStage(rnnStage_, [this, &embFuture, tokens, sessionIdCopy, &rnnPromise, mode]() {
                auto embedding = embFuture.get(); // 等待embedding完成
                std::string ctx;
                if (mode == "rnn")
                {
                    if (useTorchModels_)
                    {
#ifdef HAVE_TORCH
                        if (torchModels_.ready)
                        {
                            auto hidden = torchModels_.hiddenGru(tokens);
                            ctx = buildTextHintFromTorch(hidden, tokens, "rnn");
                        }
#endif
                    }
                    if (ctx.empty())
                    {
                        std::vector<float> rnnHidden;
                        {
                            std::lock_guard<std::mutex> lock(mu_);
                            auto &state = sessions_[sessionIdCopy];
                            if (state.rnnHidden.empty())
                            {
                                state.rnnHidden.assign(rnn_.hiddenDim, 0.0f);
                            }
                            rnnHidden = state.rnnHidden;
                        }
                        auto newHidden = rnn_.step(embedding, rnnHidden);
                        {
                            std::lock_guard<std::mutex> lock(mu_);
                            auto &state = sessions_[sessionIdCopy];
                            state.rnnHidden = newHidden;
                        }
                        ctx = buildTextHint(newHidden, tokens, "rnn");
                    }
                }
                else if (mode == "lstm")
                {
                    if (useTorchModels_)
                    {
#ifdef HAVE_TORCH
                        if (torchModels_.ready)
                        {
                            auto hidden = torchModels_.hiddenLstm(tokens);
                            ctx = buildTextHintFromTorch(hidden, tokens, "lstm");
                        }
#endif
                    }
                    if (ctx.empty())
                    {
                        std::vector<float> lstmHidden, lstmCell;
                        {
                            std::lock_guard<std::mutex> lock(mu_);
                            auto &state = sessions_[sessionIdCopy];
                            if (state.lstmHidden.empty())
                            {
                                state.lstmHidden.assign(lstm_.hiddenDim, 0.0f);
                                state.lstmCell.assign(lstm_.hiddenDim, 0.0f);
                            }
                            lstmHidden = state.lstmHidden;
                            lstmCell = state.lstmCell;
                        }
                        auto next = lstm_.step(embedding, lstmHidden, lstmCell);
                        std::vector<float> newLstmHidden, newLstmCell;
                        {
                            std::lock_guard<std::mutex> lock(mu_);
                            auto &state = sessions_[sessionIdCopy];
                            state.lstmHidden = std::move(next.first);
                            state.lstmCell = std::move(next.second);
                            newLstmHidden = state.lstmHidden;
                            newLstmCell = state.lstmCell;
                        }
                        ctx = buildTextHint(newLstmHidden, tokens, "lstm");
                    }
                }
                rnnPromise.set_value(ctx);
            });

            // 阶段3: 上下文整合（异步，依赖RNN/LSTM结果）
            std::promise<std::string> contextPromise;
            std::future<std::string> contextFuture = contextPromise.get_future();
            submitToStage(contextStage_, [this, &rnnFuture, shortContext, lexicalContext, mode, &contextPromise]() {
                auto rnnContext = rnnFuture.get(); // 等待RNN/LSTM完成
                std::string context;
                if (mode == "rnn" || mode == "lstm")
                {
                    context = buildHybridContext(shortContext, lexicalContext, rnnContext);
                }
                else
                {
                    context = shortContext.empty() ? lexicalContext : shortContext;
                }
                contextPromise.set_value(context);
            });

            // 等待所有阶段完成
            auto embedding = embFuture.get();
            auto context = contextFuture.get();

            state.lastMode = mode;
            Json::Value out;
            out["sessionId"] = sessionId;
            out["mode"] = mode;
            out["tokenCount"] = tokenCount;
            out["messageCount"] = state.messageCount;
            out["context"] = context;
            return out;
        }

        // 为 /api/chat 准备会话上下文（核心：把 ContextService 接入主聊天流）。
        // 调用方式：前端代理在转发聊天请求前调用，传入当前用户消息。
        // 实现思路：推进会话计数 → 按 messageCount 路由 concat(<=5)/rnn(<=15)/lstm(>15)
        //          → 推入短期窗口 → 渲染 "用户:/助手:" 多轮历史（近距离传统上下文）
        //          → 中长程附加 RNN/LSTM 压缩语义摘要行
        //          → 检索 Episodic Memory 注入跨 session 相关摘要。
        // 返回值：可直接作为 contextHint 注入的字符串；空表示无可用上下文。
        // 注意事项：返回的历史包含当前这条用户消息（gateway 会去重末尾 user 行）。
        std::string prepareChatContext(const std::string &sessionId, const std::string &text, const std::string &modeHint, float latencyMs = 0.0f)
        {
            ensureTorchReady();
            std::lock_guard<std::mutex> lock(mu_);
            auto &state = sessions_[sessionId];
            state.messageCount += 1;

            // 将 AdaptiveController 的动态 maxMessages 应用到 shortWindow
            if (state.adaptive.maxMessages > 0)
                state.shortWindow.maxMessages = static_cast<size_t>(state.adaptive.maxMessages);

            const auto tokens = Tokenizer::tokenize(text);
            const int tokenCount = static_cast<int>(tokens.size());
            const std::string mode = selectMode(modeHint, tokenCount, state);

            auto embedding = embeddings_.embedText(text);
#ifdef HAVE_TORCH
            if (useTorchModels_ && torchModels_.ready)
            {
                embedding = torchModels_.embedText(tokens);
            }
#endif

            std::string adaptiveHint;
            if (mode == "rnn")
            {
                if (state.rnnHidden.empty())
                    state.rnnHidden.assign(rnn_.hiddenDim, 0.0f);
                state.rnnHidden = rnn_.step(embedding, state.rnnHidden);
                adaptiveHint = buildTextHint(state.rnnHidden, tokens, "rnn");
            }
            else if (mode == "lstm")
            {
                if (state.lstmHidden.empty())
                {
                    state.lstmHidden.assign(lstm_.hiddenDim, 0.0f);
                    state.lstmCell.assign(lstm_.hiddenDim, 0.0f);
                }
                auto next = lstm_.step(embedding, state.lstmHidden, state.lstmCell);
                state.lstmHidden = std::move(next.first);
                state.lstmCell = std::move(next.second);
                adaptiveHint = buildTextHint(state.lstmHidden, tokens, "lstm");
            }
            state.lastMode = mode;
            // 用实测延迟更新 AdaptiveController（下轮生效）
            if (latencyMs > 0.0f)
                state.adaptive.update(latencyMs);

            // 检索 Episodic Memory（跨 session 相关摘要）
            std::string episodicHint;
            if (state.messageCount == 1) // 仅在会话首次检索，避免重复
            {
                episodicHint = retrieveEpisodicMemory(text);
            }

            // 渲染 "用户:/助手:" 多轮历史（applyContextHintToText 会整体前置给 LLM；
            // 同时 gateway 094/093 解析器可将其拆为多轮 user/assistant 消息）。
            // 长对话优化：concat/rnn/lstm 均渲染完整 shortWindow（shortWindow 自身按 maxTokens 裁剪）。
            // 更早历史由 RNN/LSTM 压缩摘要 adaptiveHint 承担，避免 prompt 过大导致 CPU 推理超时。
            std::ostringstream ss;
            ss << "[Conversation history:\n"; // [本次对话历史：
            bool anyTurn = false;
            size_t startIdx = 0;
            // 修复：不再重复拼接 concat.history 和 shortWindow 的内容。
            // shortWindow 已经包含完整的 user/assistant 交替，concat.history 的额外遍历会导致用户消息重复。
            {
                size_t idx = 0;
                for (const auto &m : state.shortWindow.messages)
                {
                    size_t cur = idx++;
                    if (cur < startIdx)
                        continue;
                    if (m.content.empty())
                        continue;
                    if (m.role == "user")
                    {
                        ss << "  User: " << m.content << "\n"; // 用户:
                        anyTurn = true;
                    }
                    else if (m.role == "assistant")
                    {
                        ss << "  Assistant: " << m.content << "\n"; // 助手:
                        anyTurn = true;
                    }
                }
            }
            if (!adaptiveHint.empty())
                ss << "  [Long-term memory summary|" << mode << "]: " << adaptiveHint << "\n"; // 长期记忆摘要
            if (!episodicHint.empty())
                ss << "  [Cross-session history]:\n" << episodicHint; // 跨会话历史
            ss << "]";
            std::cout << "[prepareChatContext] Final: anyTurn=" << anyTurn << ", episodicHint.empty=" << episodicHint.empty() << ", willReturn=" << (!anyTurn && episodicHint.empty() ? "false" : "true") << std::endl;
            if (!anyTurn && episodicHint.empty())
            {
                // Even though there is no previous history to inject as a hint, we must still
                // record this first user message so the next turn can see it.
                state.shortWindow.pushUser(text);
                state.concat.push("User: " + text);
                std::cout << "[prepareChatContext] Returning empty string" << std::endl;
                return std::string();
            }
            std::string result = ss.str();
            std::cout << "[prepareChatContext] Returning context with length=" << result.length() << std::endl;

            // 当前用户消息在上下文构建完成后才推入会话状态，供下一轮使用。
            state.shortWindow.pushUser(text);
            state.concat.push("User: " + text);
            return result;
        }

        // 记录助手回复到会话短期窗口与词面历史。
        // 调用方式：前端代理收到上游回复后调用，使下一轮能看到本轮回答。
        // 实现思路：把回复推入 shortWindow（assistant 角色）和 concat。
        // 注意事项：仅当会话已存在（prepareChatContext 创建过）时才记录。
        void recordAssistantReply(const std::string &sessionId, const std::string &reply)
        {
            if (reply.empty())
                return;
            std::lock_guard<std::mutex> lock(mu_);
            auto it = sessions_.find(sessionId);
            if (it == sessions_.end())
                return;
            it->second.shortWindow.pushAssistant(reply);
            it->second.concat.push("Assistant: " + reply); // 助手：统一为英文前缀，避免编码和混合格式问题
        }

        // 剥离事实陈述常见前缀（"请记住："/"记住："等），得到可用于探测的主体内容。
        static std::string stripFactPrefix(const std::string &fact)
        {
            static const char *prefixes[] = {
                "\xe8\xaf\xb7\xe8\xae\xb0\xe4\xbd\x8f\xef\xbc\x9a", // 请记住：
                "\xe8\xaf\xb7\xe8\xae\xb0\xe4\xbd\x8f",             // 请记住
                "\xe8\xae\xb0\xe4\xbd\x8f\xef\xbc\x9a",             // 记住：
                "\xe8\xae\xb0\xe4\xbd\x8f",                         // 记住
                "\xe8\xaf\xb7\xe4\xbd\xa0\xe8\xae\xb0\xe4\xbd\x8f", // 请你记住
            };
            std::string s = fact;
            // 去除首部空白
            size_t b = s.find_first_not_of(" \t\r\n");
            if (b != std::string::npos)
                s = s.substr(b);
            for (const char *p : prefixes)
            {
                std::string pre(p);
                if (s.size() >= pre.size() && s.compare(0, pre.size(), pre) == 0)
                {
                    s = s.substr(pre.size());
                    size_t b2 = s.find_first_not_of(" \t\r\n\xef\xbc\x9a:");
                    if (b2 != std::string::npos)
                        s = s.substr(b2);
                    break;
                }
            }
            return s;
        }

        // 检测回答中是否包含明显的否定/反义词语（表示模型不了解或与事实相悖）。
        static bool hasNegation(const std::string &reply)
        {
            std::string low = reply;
            std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            static const char *markers[] = {
                "\xe4\xb8\x8d\xe4\xba\x86\xe8\xa7\xa3", // 不了解
                "\xe4\xb8\x8d\xe7\x9f\xa5\xe9\x81\x93", // 不知道
                "\xe4\xb8\x8d\xe6\xb8\x85\xe6\xa5\x9a", // 不清楚
                "\xe4\xb8\x8d\xe7\xa1\xae\xe5\xae\x9a", // 不确定
                "\xe6\x97\xa0\xe6\xb3\x95",             // 无法
                "\xe6\xb2\xa1\xe6\x9c\x89\xe7\x9b\xb8\xe5\x85\xb3", // 没有相关
                "\xe6\x97\xa0\xe7\x9b\xb8\xe5\x85\xb3", // 无相关
                "\xe6\x97\xa0\xe6\xb3\x95\xe7\xa1\xae\xe5\xae\x9a", // 无法确定
                "\xe6\x8a\xb1\xe6\xad\x89",             // 抱歉
                "don't", "do not", "cannot", "can't", "not aware",
                "no information", "not sure", "unknown", "i'm not", "i am not",
            };
            for (const char *m : markers)
            {
                if (low.find(m) != std::string::npos)
                    return true;
            }
            return false;
        }

        // 两个向量的余弦相似度。
        static float cosineOf(const std::vector<float> &a, const std::vector<float> &b)
        {
            if (a.empty() || b.empty() || a.size() != b.size())
                return 0.0f;
            float dot = 0.0f, na = 0.0f, nb = 0.0f;
            for (size_t i = 0; i < a.size(); ++i)
            {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            float n = std::sqrt(na) * std::sqrt(nb);
            return n > 1e-9f ? dot / n : 0.0f;
        }

        // 探测基座 LLM 是否"已知"某事实。
        // 实现思路：向 llamacpp 适配器发不带上下文的探测问句，比较回答与事实的余弦相似度，
        //          并检测否定/反义词语；相似度过低或含否定词 → 判定为"未知"（返回 false）。
        // 注意事项：探测失败（超时/网络错误）时返回 false（按未知处理，倾向保留信息）。
        // 注意事项：该方法发起阻塞式 HTTP，应在后台线程调用，不可占用会话锁。
        bool probeBaseModelKnows(const std::string &fact)
        {
            std::string subject = stripFactPrefix(fact);
            if (subject.empty())
                subject = fact;
            try
            {
                std::string selectedModel = probeModel_.empty() ? std::string("llamacpp") : probeModel_;
                std::string prompt = subject +
                    "\n\n\xef\xbc\x88\xe8\xaf\xb7\xe4\xbb\x85\xe6\xa0\xb9\xe6\x8d\xae\xe4\xbd\xa0\xe5\xb7\xb2\xe6\x9c\x89\xe7\x9a\x84\xe7\x9f\xa5\xe8\xaf\x86\xe7\xae\x80\xe7\x9f\xad\xe5\x9b\x9e\xe7\xad\x94\xef\xbc\x9b\xe5\xa6\x82\xe6\x9e\x9c\xe4\xbd\xa0\xe5\xb9\xb6\xe4\xb8\x8d\xe4\xba\x86\xe8\xa7\xa3\xe4\xb8\x8a\xe8\xbf\xb0\xe4\xbf\xa1\xe6\x81\xaf\xef\xbc\x8c\xe8\xaf\xb7\xe7\x9b\xb4\xe6\x8e\xa5\xe5\x9b\x9e\xe7\xad\x94\xef\xbc\x9a\xe4\xb8\x8d\xe4\xba\x86\xe8\xa7\xa3\xe3\x80\x82\xef\xbc\x89"; // （请仅根据你已有的知识简短回答；如果你并不了解上述信息，请直接回答：不了解。）
                nlohmann::json payload = {
                    {"model", selectedModel},
                    {"stream", false},
                    {"messages", nlohmann::json::array({nlohmann::json{{"role", "user"}, {"content", prompt}}})},
                    {"max_tokens", 96}};

                auto client = drogon::HttpClient::newHttpClient(probeBaseUrl_);
                if (!client)
                    return false;
                auto req = drogon::HttpRequest::newHttpRequest();
                req->setMethod(drogon::Post);
                req->setPath("/v1/chat/completions");
                req->setContentTypeCode(drogon::CT_APPLICATION_JSON);
                req->setBody(payload.dump());

                std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>> promise;
                auto future = promise.get_future();
                client->sendRequest(req, [&promise](drogon::ReqResult result, const drogon::HttpResponsePtr &resp)
                                    { try { promise.set_value({result, resp}); } catch (...) {} });
                if (future.wait_for(std::chrono::milliseconds(std::max(2000, probeTimeoutMs_))) != std::future_status::ready)
                    return false; // 超时按未知处理
                auto pr = future.get();
                if (pr.first != drogon::ReqResult::Ok || !pr.second)
                    return false;
                if (pr.second->statusCode() < 200 || pr.second->statusCode() >= 300)
                    return false;
                nlohmann::json doc = nlohmann::json::parse(std::string(pr.second->getBody()), nullptr, false);
                if (doc.is_discarded())
                    return false;
                std::string reply;
                // Try OpenAI-compatible format (llama-server v1)
                if (doc.contains("choices") && doc["choices"].is_array() && doc["choices"].size() > 0)
                {
                    auto& choice = doc["choices"][0];
                    if (choice.contains("message") && choice["message"].is_object() && choice["message"].contains("content") && choice["message"]["content"].is_string())
                        reply = choice["message"]["content"].get<std::string>();
                }
                // Fallback to legacy format
                if (reply.empty())
                {
                    if (doc.contains("message") && doc["message"].is_object() && doc["message"].contains("content") && doc["message"]["content"].is_string())
                        reply = doc["message"]["content"].get<std::string>();
                    else if (doc.contains("response") && doc["response"].is_string())
                        reply = doc["response"].get<std::string>();
                }
                if (reply.empty())
                    return false;
                if (hasNegation(reply))
                    return false; // 明显否定/反义 → 未知
                float sim = cosineOf(embeddings_.embedText(reply), embeddings_.embedText(subject));
                return sim >= knownSimThreshold_; // 相似度达标 → 已知；否则未知
            }
            catch (...)
            {
                return false;
            }
        }

        // 后台跨 session 学习：探测候选事实，仅将"未知"事实持久化到 Episodic Memory。
        // 调用方式：reset 中以 detached 线程启动，不阻塞会话清理。
        // 实现思路：逐条探测基座 LLM；已知 → 隔离（丢弃），未知 → 归档为跨 session 记忆。
        void learnUnknownFacts(const std::string &sessionId, std::vector<std::string> facts)
        {
            try
            {
                learnInFlight_.fetch_add(1);
                std::cout << "[cross-session] learnUnknownFacts started for session " << sessionId << " with " << facts.size() << " facts" << std::endl;
                std::vector<std::string> unknownFacts;
                for (const auto &f : facts)
                {
                    if (f.empty())
                        continue;
                    try
                    {
                        std::cout << "[cross-session] Probing fact: " << f << std::endl;
                        bool known = probeBaseModelKnows(f);
                        std::cout << "[cross-session] Fact '" << f << "' known=" << known << std::endl;
                        if (!known)
                            unknownFacts.push_back(f);
                    }
                    catch (...)
                    {
                        std::cout << "[cross-session] Probe failed for fact: " << f << ", treating as unknown" << std::endl;
                        // 探测失败时按未知处理，倾向保留信息
                        unknownFacts.push_back(f);
                    }
                }
                std::cout << "[cross-session] Found " << unknownFacts.size() << " unknown facts out of " << facts.size() << std::endl;
                if (!unknownFacts.empty())
                {
                    std::string summary;
                    for (const auto &f : unknownFacts)
                    {
                        if (!summary.empty())
                            summary += "; ";
                        summary += stripFactPrefix(f);
                    }
                    EpisodicMemoryEntry entry;
                    entry.sessionId = sessionId;
                    entry.summary = summary;
                    entry.facts = unknownFacts;
                    entry.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
                    entry.relevanceScore = 0.0f;
                    {
                        std::lock_guard<std::mutex> lock(mu_);
                        try
                        {
                            entry.embedding = embeddings_.embedText(summary);
                            if (episodicMemory_.size() >= 100)
                                episodicMemory_.erase(episodicMemory_.begin());
                            episodicMemory_.push_back(std::move(entry));
                        }
                        catch (...)
                        {
                            // embedding失败时跳过，不影响主流程
                        }
                    }
                    learnedFactCount_.fetch_add(static_cast<int>(unknownFacts.size()));
                }
            }
            catch (...)
            {
                // 整体异常处理，确保计数器正确
            }
            learnInFlight_.fetch_sub(1);
        }

        // 计算查询与目标文本的词面重叠分（query 中有多少比例的内容词出现在目标里）。
        // 用于捕捉专有名词/工号等 OOV token——它们的 embedding 近零，余弦无法匹配，
        // 但词面重叠能可靠匹配跨 session 学习到的未知事实。
        static float lexicalOverlap(const std::string &query, const std::string &target)
        {
            auto qTokens = Tokenizer::tokenize(query);
            auto tTokens = Tokenizer::tokenize(target);
            if (qTokens.empty() || tTokens.empty())
                return 0.0f;
            std::unordered_set<std::string> tset(tTokens.begin(), tTokens.end());
            std::unordered_set<std::string> qseen;
            int hit = 0, total = 0;
            for (const auto &qt : qTokens)
            {
                if (qt.size() < 2) // 跳过过短 token（虚词/标点）
                    continue;
                if (!qseen.insert(qt).second)
                    continue; // 去重
                total++;
                if (tset.count(qt))
                    hit++;
            }
            return total > 0 ? static_cast<float>(hit) / static_cast<float>(total) : 0.0f;
        }

        std::string retrieveEpisodicMemory(const std::string &query)
        {
            if (episodicMemory_.empty())
                return std::string();

            auto queryEmbedding = embeddings_.embedText(query);

            // 综合评分：embedding 余弦 与 词面重叠 取较大者，再乘时间衰减。
            std::vector<std::pair<float, size_t>> scored;
            for (size_t i = 0; i < episodicMemory_.size(); ++i)
            {
                float cosSim = queryEmbedding.empty() ? 0.0f : episodicMemory_[i].cosineSimilarity(queryEmbedding);
                // 词面重叠：对 summary 与每条 fact 取最大重叠
                float lex = lexicalOverlap(query, episodicMemory_[i].summary);
                for (const auto &f : episodicMemory_[i].facts)
                    lex = std::max(lex, lexicalOverlap(query, f));
                float score = std::max(cosSim, lex);
                // 时间衰减：24 小时内权重 1.0，超过则线性衰减
                int64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
                int64_t ageHours = (now - episodicMemory_[i].timestamp) / 3600000000; // 转换为小时
                float timeDecay = ageHours < 24 ? 1.0f : std::max(0.0f, 1.0f - (ageHours - 24) / 168.0f); // 7 天后衰减到 0
                scored.push_back({score * timeDecay, i});
            }

            // 按综合分降序排序
            std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b)
                      { return a.first > b.first; });

            // 取 top-3 相关条目（阈值 0.3：余弦或词面重叠任一达标即可）
            std::ostringstream ss;
            int count = 0;
            for (const auto &item : scored)
            {
                if (count >= 3 || item.first < 0.3f)
                    break;
                const auto &entry = episodicMemory_[item.second];
                ss << "[\xe5\x8e\x86\xe5\x8f\xb2\xe4\xbc\x9a\xe8\xaf\x9d\xe6\x91\x98\xe8\xa6\x81]: " << entry.summary << "\n"; // [历史对话摘要]
                for (const auto &fact : entry.facts)
                {
                    ss << "  - " << fact << "\n";
                }
                count++;
            }

            return count > 0 ? ss.str() : std::string();
        }

        bool reset(const std::string &sessionId)
        {
            std::vector<std::string> candidateFacts;
            int messageCount = 0;
            {
                std::lock_guard<std::mutex> lock(mu_);
                auto it = sessions_.find(sessionId);
                if (it == sessions_.end())
                {
                    std::cout << "[cross-session] reset: session not found: " << sessionId << std::endl;
                    return false;
                }
                messageCount = it->second.messageCount;
                std::cout << "[cross-session] reset: session found, messageCount=" << messageCount << std::endl;
                // 修复：提取候选事实时，同时从shortWindow和concat.history中提取
                // 确保concat模式下的历史消息也能用于跨会话学习
                int factCount = 0;
                // 先从concat.history中提取（包含早期消息）
                for (const auto &hist : it->second.concat.history)
                {
                    // 只提取用户消息（以"用户:"开头的）
                    if (hist.find("User: ") == 0 && factCount < 5)
                    {
                        std::string fact = hist.substr(6); // 去掉"用户: "前缀
                        // 过滤掉非事实性消息（如"tell me about topic X"）
                        if (fact.find("tell me about") == std::string::npos &&
                            fact.find("Tell me about") == std::string::npos &&
                            fact.find("What is") == std::string::npos)
                        {
                            candidateFacts.push_back(fact);
                            factCount++;
                        }
                    }
                }
                // 再从shortWindow中提取（最近消息）
                for (const auto &msg : it->second.shortWindow.messages)
                {
                    if (msg.role == "user" && factCount < 5)
                    {
                        // 过滤掉非事实性消息
                        if (msg.content.find("tell me about") == std::string::npos &&
                            msg.content.find("Tell me about") == std::string::npos &&
                            msg.content.find("What is") == std::string::npos)
                        {
                            candidateFacts.push_back(msg.content);
                            factCount++;
                        }
                    }
                }
                std::cout << "[cross-session] reset: extracted " << candidateFacts.size() << " candidate facts (from concat.history and shortWindow)" << std::endl;
                sessions_.erase(it);
            }
            // 锁外异步探测：仅将基座 LLM"未知"的事实持久化为跨 session 记忆，"已知"的隔离丢弃。
            std::cout << "[cross-session] reset: crossSessionLearnEnabled=" << crossSessionLearnEnabled_ << ", messageCount=" << messageCount << ", candidateFacts.size()=" << candidateFacts.size() << std::endl;
            if (crossSessionLearnEnabled_ && messageCount >= 1 && !candidateFacts.empty())
            {
                std::cout << "[cross-session] reset: launching learnUnknownFacts thread" << std::endl;
                std::thread(&ContextService::learnUnknownFacts, this, sessionId, candidateFacts).detach();
            }
            else
            {
                std::cout << "[cross-session] reset: conditions not met, skipping learnUnknownFacts" << std::endl;
            }
            return true;
        }

        Json::Value status(const std::string &sessionId)
        {
            std::lock_guard<std::mutex> lock(mu_);
            Json::Value out;
            auto it = sessions_.find(sessionId);
            if (it == sessions_.end())
            {
                out["exists"] = false;
                return out;
            }
            out["exists"] = true;
            out["sessionId"] = sessionId;
            out["lastMode"] = it->second.lastMode;
            out["messageCount"] = it->second.messageCount;
            out["shortWindowItems"] = static_cast<int>(it->second.shortWindow.messages.size());
            out["concatItems"] = static_cast<int>(it->second.concat.history.size());
            return out;
        }

        // 跨 session 学习全局状态：供测试轮询与运行监控。
        Json::Value knowledgeStatus()
        {
            Json::Value out;
            out["crossSessionLearnEnabled"] = crossSessionLearnEnabled_;
            out["learnedFactCount"] = learnedFactCount_.load();
            out["learnInFlight"] = learnInFlight_.load();
            {
                std::lock_guard<std::mutex> lock(mu_);
                out["episodicEntries"] = static_cast<int>(episodicMemory_.size());
            }
            out["knownSimThreshold"] = knownSimThreshold_;
            return out;
        }

        static std::string generateSessionId()
        {
            auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            std::ostringstream ss;
            ss << "s_" << std::hex << now;
            return ss.str();
        }

    private:
        static std::string truncateContextText(const std::string &value, size_t limit)
        {
            if (value.size() <= limit)
            {
                return value;
            }
            if (limit <= 3)
            {
                return value.substr(0, limit);
            }
            return value.substr(0, limit - 3) + "...";
        }

        static std::string buildHybridContext(const std::string &shortContext,
                                              const std::string &lexicalContext,
                                              const std::string &adaptiveHint)
        {
            std::ostringstream ss;
            if (!shortContext.empty())
            {
                ss << "[short_term]\n" << truncateContextText(shortContext, 1200) << "\n";
            }
            if (!lexicalContext.empty())
            {
                ss << "[long_term_lexical]\n" << truncateContextText(lexicalContext, 700) << "\n";
            }
            if (!adaptiveHint.empty())
            {
                ss << "[long_term_adaptive]\n" << truncateContextText(adaptiveHint, 220) << "\n";
            }
            return ss.str();
        }

        std::string selectMode(const std::string &modeHint, int tokenCount, const SessionState &state) const
        {
            if (modeHint == "rnn" || modeHint == "lstm" || modeHint == "concat")
            {
                return modeHint;
            }
            (void)tokenCount;
            // 动态阈值：优先使用每 session 的 AdaptiveController 值（经实测延迟调整）
            // 若 adaptive 尚未运行（roundCount==0），则使用全局初始值（once_flag，config/env 可覆盖）。
            static int globalConcatThresh = 5;
            static int globalRnnThresh = 15;
            static std::once_flag threshInit;
            std::call_once(threshInit, []() {
                globalConcatThresh = std::max(1, resolveConfig<int>("context.adaptive.concatThresh", 5, "FRONTEND_CONCAT_THRESH"));
                globalRnnThresh = std::max(globalConcatThresh + 1, resolveConfig<int>("context.adaptive.rnnThresh", 15, "FRONTEND_RNN_THRESH"));
            });
            int concatThresh = (state.adaptive.roundCount > 0) ? state.adaptive.concatThresh : globalConcatThresh;
            int rnnThresh = (state.adaptive.roundCount > 0) ? state.adaptive.rnnThresh : globalRnnThresh;
            if (state.messageCount <= concatThresh)
            {
                return "concat";
            }
            if (state.messageCount <= rnnThresh)
            {
                return "rnn";
            }
            return "lstm";
        }

        std::string buildTextHint(const std::vector<float> &hidden,
                                  const std::vector<std::string> &tokens,
                                  const std::string &tag)
        {
            if (tokens.empty())
            {
                return "[" + tag + "]";
            }
            double hNorm = 0.0;
            for (float v : hidden)
                hNorm += v * v;
            hNorm = std::sqrt(std::max(1e-9, hNorm));

            // 优化：限制top-k候选词数量，避免遍历全词表
            static int topKLimit = 50;
            static std::once_flag topKInit;
            std::call_once(topKInit, []() {
                topKLimit = std::max(10, resolveConfig<int>("context.rnnTopK", 50, "FRONTEND_RNN_TOP_K"));
            });

            // ScoredToken 描述 token 与相似度评分的配对项。
            // 调用方式：用于文本提示词筛选阶段的排序容器。
            // 实现思路：保存词项和计算出的相似度分值。
            // 注意事项：score 越大表示与当前隐藏状态越相关。
            // 注意事项：结构体仅在函数局部短期使用。
            // 注意事项：排序前需先去重，避免重复词干扰。
            struct ScoredToken
            {
                std::string token;
                double score{0.0};
            };
            std::vector<ScoredToken> scored;
            // 修复：使用整个词汇表而非仅当前 tokens，让隐藏状态真正输出“主题/摘要”词汇。
            scored.reserve(embeddings_.idToWord.size());
            std::unordered_set<std::string> seen;
            for (const auto &t : embeddings_.idToWord)
            {
                if (t.size() <= 1 || seen.count(t))
                    continue;
                seen.insert(t);
                auto vec = embeddings_.get(t);
                if (vec.empty())
                    continue; // 跳过不在词汇表中的token
                double dot = 0.0;
                double vNorm = 0.0;
                for (size_t i = 0; i < vec.size() && i < hidden.size(); i++)
                {
                    dot += vec[i] * hidden[i];
                    vNorm += vec[i] * vec[i];
                }
                vNorm = std::sqrt(std::max(1e-9, vNorm));
                double sim = dot / (hNorm * vNorm);
                scored.push_back({t, sim});
            }
            std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b)
                      { return a.score > b.score; });
            // 优化：只保留top-k候选词，而非全部
            size_t keep = std::min<size_t>(static_cast<size_t>(topKLimit), scored.size());
            // 构建关键词字符串（用于 summary model 输入 和 fallback 输出）
            std::ostringstream kwSs;
            for (size_t i = 0; i < keep; i++)
                kwSs << (i == 0 ? "" : " ") << scored[i].token;
            if (keep == 0)
            {
                size_t fallbackCount = std::min<size_t>(3, tokens.size());
                for (size_t i = 0; i < fallbackCount; i++)
                    kwSs << (i == 0 ? "" : " ") << tokens[i];
            }
            const std::string keywords = kwSs.str();
            // 尝试用 seq2seq 组句模型将关键词云转换为自然语言摘要句子
            if (summaryModel_.ready && !summaryModel_.failed)
            {
                std::string sentence = summaryModel_.generateSummary(keywords);
                if (!sentence.empty())
                    return "[" + tag + "] " + sentence;
            }
            // Fallback: 关键词云（原始行为）
            return "[" + tag + "] " + keywords;
        }

#ifdef HAVE_TORCH
        std::string buildTextHintFromTorch(const torch::Tensor &hidden,
                                           const std::vector<std::string> &tokens,
                                           const std::string &tag)
        {
            if (tokens.empty() || !torchModels_.ready)
                return "[" + tag + "]";
            // 修复：使用对应模型的输出头从 hidden 生成词汇分布，而不是对当前输入 token 做相似度。
            auto model = (tag == "lstm") ? torchModels_.lstm : torchModels_.gru;
            if (!model)
                return "[" + tag + "]";
            torch::NoGradGuard guard;
            auto logits = model->fc->forward(hidden).squeeze(0); // [vocab]
            auto probs = torch::softmax(logits, 0);
            constexpr int kTop = 8;
            auto topk = torch::topk(probs, kTop, 0);
            auto indices = std::get<1>(topk).accessor<int64_t, 1>();
            std::ostringstream ss;
            ss << "[" << tag << "]";
            for (int i = 0; i < kTop; i++)
            {
                int64_t id = indices[i];
                if (id < 0 || id >= (int64_t)torchModels_.vocab.itos.size())
                    continue;
                const std::string &t = torchModels_.vocab.itos[(size_t)id];
                if (t.empty() || t == "<pad>" || t == "<unk>")
                    continue;
                ss << " " << t;
            }
            return ss.str();
        }
#endif

        // ---------------------------------------------------------------
        // SummaryModel: 小型 seq2seq transformer，将 top-k 关键词云组成
        // 完整自然语言摘要句子注入 contextHint。
        // 预训练语料：robots/wikitext-103-all.txt（自回归 next-token prediction）
        // 微调任务：top-k 词序列 -> 对话末轮首句（伪摘要标签，无需人工标注）
        // 推理：buildTextHint 调用，若模型未就绪则 fallback 关键词云。
        // ---------------------------------------------------------------
        struct SummaryModel
        {
            transformer::TransformerParams params;
            std::unique_ptr<transformer::TransformerModel> model;
            transformer::Tokenizer tokenizer{phoenix::cfgOr<int>("summary_model.vocabSize", 32000)};
            std::atomic<bool> ready{false};
            std::atomic<bool> failed{false};
            std::mutex mu;
            std::string checkpointPath;
            int maxSummaryTokens{32};  // 环境变量 FRONTEND_SUMMARY_MAX_TOKENS 控制，初始值 32，后期实验确定

            SummaryModel()
            {
                params.vocabSize =
                    phoenix::cfgOr<int>("summary_model.vocabSize", 32000);
                params.dModel =
                    phoenix::cfgOr<int>("summary_model.dModel", 512);
                params.nHeads =
                    phoenix::cfgOr<int>("summary_model.nHeads", 8);
                params.nLayers =
                    phoenix::cfgOr<int>("summary_model.nLayers", 6);
                params.dFF =
                    phoenix::cfgOr<int>("summary_model.dFF", 2048);
                params.maxLen =
                    phoenix::cfgOr<int>("summary_model.maxLen", 2048);
                params.maxTokens =
                    phoenix::cfgOr<int>("summary_model.maxTokens", 1024);
                params.lr =
                    phoenix::cfgOr<float>("summary_model.lr", 0.001f);
                params.tokenizerMode =
                    phoenix::cfgOr<std::string>("summary_model.tokenizerMode", std::string("bpe"));
                maxSummaryTokens = std::max(8, resolveConfig<int>("summary_model.maxTokens", 32, "FRONTEND_SUMMARY_MAX_TOKENS"));
                checkpointPath = resolveConfig<std::string>("summary_model.checkpointPath", std::string("runtime_store/summary_model.json"), "FRONTEND_SUMMARY_CHECKPOINT");
            }

            bool tryLoadCheckpoint()
            {
                try
                {
                    std::ifstream f(checkpointPath);
                    if (!f.good()) return false;
                    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                    auto j = nlohmann::json::parse(s);
                    std::lock_guard<std::mutex> lk(mu);
                    if (!model) model = std::make_unique<transformer::TransformerModel>(params);
                    std::string err;
                    if (model->loadStateDict(j, err))
                    {
                        ready = true;
                        std::cout << "[summary] loaded checkpoint from " << checkpointPath << std::endl;
                        return true;
                    }
                }
                catch (...) {}
                return false;
            }

            void saveCheckpoint()
            {
                try
                {
                    fs::path cp(checkpointPath);
                    fs::create_directories(cp.parent_path());
                    std::lock_guard<std::mutex> lk(mu);
                    if (!model) return;
                    auto j = model->stateDict();
                    std::ofstream f(checkpointPath);
                    f << j.dump();
                    std::cout << "[summary] saved checkpoint to " << checkpointPath << std::endl;
                }
                catch (...) {}
            }

            // 从 wikitext 文件预训练（自回归 LM），再用伪摘要样本微调 seq2seq 组句能力。
            void trainAsync(const fs::path &wikitextPath, const std::vector<std::string> &dialogueSamples,
                            int pretrainLines, int epochs)
            {
                if (ready || failed) return;
                std::cout << "[summary] starting async pretraining from " << wikitextPath << std::endl;
                {
                    std::lock_guard<std::mutex> lk(mu);
                    if (!model) model = std::make_unique<transformer::TransformerModel>(params);
                }
                // 预训练：从 wikitext 读取 pretrainLines 行，做 next-token prediction
                try
                {
                    std::ifstream wf(wikitextPath);
                    if (!wf.good())
                    {
                        std::cerr << "[summary] wikitext not found: " << wikitextPath << std::endl;
                        failed = true;
                        return;
                    }
                    std::vector<transformer::TrainSample> pretrain;
                    pretrain.reserve(static_cast<size_t>(pretrainLines));
                    std::string line;
                    int lineCount = 0;
                    while (std::getline(wf, line) && lineCount < pretrainLines)
                    {
                        line.erase(0, line.find_first_not_of(" \t\r\n"));
                        if (line.size() < 20) continue;
                        // 自回归：输入=前 N/2 字符，目标=后 N/2 字符
                        size_t mid = line.size() / 2;
                        transformer::TrainSample s;
                        s.input = line.substr(0, mid);
                        s.target = line.substr(mid);
                        s.graph = "";
                        pretrain.push_back(std::move(s));
                        lineCount++;
                        // 让 tokenizer 观察语料（BPE 训练）
                        tokenizer.observe(line);
                    }
                    std::cout << "[summary] pretrain samples: " << pretrain.size() << std::endl;
                    // 实际训练
                    float lr = params.lr;
                    for (int ep = 0; ep < epochs && !failed; ep++)
                    {
                        float totalLoss = 0.0f;
                        size_t count = 0;
                        for (auto &s : pretrain)
                        {
                            auto inputToks = tokenizer.encode(s.input, params.maxLen, true);
                            auto targetToks = tokenizer.encode(s.target, params.maxLen, false);
                            if (inputToks.empty() || targetToks.empty()) continue;
                            std::lock_guard<std::mutex> lk(mu);
                            totalLoss += model->trainOnSample(inputToks, {}, targetToks, lr);
                            count++;
                        }
                        if (count > 0)
                            std::cout << "[summary] pretrain epoch " << ep + 1 << "/" << epochs
                                      << " loss=" << totalLoss / count << std::endl;
                    }
                    // 微调：用伪摘要样本（top-k 词 -> 对话首句）做 seq2seq 微调
                    if (!dialogueSamples.empty())
                    {
                        std::cout << "[summary] fine-tuning on " << dialogueSamples.size() / 2 << " pseudo-summary pairs" << std::endl;
                        for (size_t i = 0; i + 1 < dialogueSamples.size() && !failed; i += 2)
                        {
                            auto inputToks = tokenizer.encode(dialogueSamples[i], params.maxLen, true);
                            auto targetToks = tokenizer.encode(dialogueSamples[i + 1], params.maxLen, false);
                            if (inputToks.empty() || targetToks.empty()) continue;
                            std::lock_guard<std::mutex> lk(mu);
                            model->trainOnSample(inputToks, {}, targetToks, params.lr * 0.1f);
                        }
                    }
                    ready = true;
                    saveCheckpoint();
                    std::cout << "[summary] model ready" << std::endl;
                }
                catch (const std::exception &ex)
                {
                    std::cerr << "[summary] training failed: " << ex.what() << std::endl;
                    failed = true;
                }
            }

            // 将 top-k 关键词序列组成自然语言摘要句子。
            // 若模型未就绪，返回空字符串（调用方 fallback 关键词云）。
            std::string generateSummary(const std::string &keywords)
            {
                if (!ready || failed || keywords.empty()) return "";
                try
                {
                    std::lock_guard<std::mutex> lk(mu);
                    if (!model) return "";
                    auto inputToks = tokenizer.encode(keywords, params.maxLen, true);
                    if (inputToks.empty()) return "";
                    auto outToks = model->generate(inputToks, {}, maxSummaryTokens, 0.0f, params.vocabSize);
                    auto result = tokenizer.decode(outToks);
                    // 截断到第一个句号（要求输出完整句子）
                    size_t dotPos = result.find_first_of(".!?");
                    if (dotPos != std::string::npos)
                        result = result.substr(0, dotPos + 1);
                    result.erase(0, result.find_first_not_of(" \t\r\n"));
                    return result;
                }
                catch (...) { return ""; }
            }
        };

        SummaryModel summaryModel_;  // 小型 seq2seq 组句模型（WikiText 预训练）

        std::mutex mu_;
        EmbeddingStore embeddings_;
        RNNModel rnn_;
        LSTMModel lstm_;
        std::unordered_map<std::string, SessionState> sessions_;
        std::vector<EpisodicMemoryEntry> episodicMemory_; // Episodic Memory 层：跨 session 对话摘要
        // 跨 session 学习配置：探测基座 LLM 判定事实"已知/未知"，仅持久化未知事实。
        std::string probeBaseUrl_{"http://127.0.0.1:8082"};
        std::string probeModel_;
        int probeTimeoutMs_{60000};
        float knownSimThreshold_{0.5f};
        bool crossSessionLearnEnabled_{true};
        std::atomic<int> learnedFactCount_{0};   // 已学习（未知）事实计数
        std::atomic<int> learnInFlight_{0};       // 正在后台探测的会话数
        bool useTorchModels_{false};
        fs::path robotsDir_;
        int maxTorchFiles_{240};
        std::once_flag torchInitOnce_;
        // 异步流水线系统：各层独立处理不阻塞
        PipelineStage embeddingStage_;    // embedding计算阶段
        PipelineStage rnnStage_;          // RNN/LSTM计算阶段
        PipelineStage contextStage_;      // 上下文整合阶段
        PipelineStage episodicStage_;     // 跨session记忆检索阶段
#ifdef HAVE_TORCH
        TorchTextModels torchModels_;
#endif

        void ensureTorchReady()
        {
#ifdef HAVE_TORCH
            if (!useTorchModels_)
                return;
            std::call_once(torchInitOnce_, [this]()
                           { torchModels_.initFromCorpus(robotsDir_, maxTorchFiles_); });
#endif
        }
    };
}

static std::vector<std::string> emotionTokenize(const std::string &text)
{
    std::vector<std::string> tokens;
    std::string cur;
    for (char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '\'') {
            cur.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        } else if (!cur.empty()) {
            tokens.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

void setupFrontendServer()
{
    ensureFrontendArena();
    const std::string webRoot = resolveConfig<std::string>("frontend_server.webRoot", std::string("./079project_frontend/build"), "WEB_ROOT");
    const std::string host = resolveConfig<std::string>("frontend_server.host", std::string("127.0.0.1"), "FRONTEND_HOST");
    const int port = resolveConfig<int>("frontend_server.port", 5081, "FRONTEND_PORT");
    const std::string robotsDir = resolveConfig<std::string>("frontend_server.robotsDir", std::string("./robots"), "ROBOTS_DIR");

    static auto boolEnv = [](const std::string &value, bool fallback)
    {
        if (value.empty())
            return fallback;
        std::string v = value;
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        return !(v == "0" || v == "false" || v == "off" || v == "no");
    };
    static auto parseWorldModelInt = [](const std::string &value, int fallback)
    {
        try
        {
            return std::stoi(value);
        }
        catch (...)
        {
            return fallback;
        }
    };
    static auto parseWorldModelDouble = [](const std::string &value, double fallback)
    {
        try
        {
            return std::stod(value);
        }
        catch (...)
        {
            return fallback;
        }
    };
    const bool useTorch = resolveConfig<bool>("frontend_server.useTorch", false, "FRONTEND_USE_TORCH");
    const fs::path worldModelDbPath = fs::path(resolveConfig<std::string>("world_model.storage.db", std::string("./runtime_store/frontend_world_model.sqlite"), "FRONTEND_WORLDMODEL_DB"));
    const fs::path worldModelLegacyDir = fs::path(resolveConfig<std::string>("world_model.storage.legacyDir", std::string("./lmdb/frontend_world_model"), "FRONTEND_WORLDMODEL_LEGACY_DIR"));
    const std::string worldModelRedisUrl = resolveConfig<std::string>("world_model.storage.redisUrl", std::string("redis://127.0.0.1:6379"), "FRONTEND_WORLDMODEL_REDIS_URL", "REDIS_URL");
    const int worldModelRedisDb = resolveConfig<int>("world_model.storage.redisDb", 7, "FRONTEND_WORLDMODEL_REDIS_DB");
    const std::string worldModelRedisPrefix = resolveConfig<std::string>("world_model.storage.redisPrefix", std::string("phoenix:frontend:world"), "FRONTEND_WORLDMODEL_REDIS_PREFIX");
    const int defaultWorldAgentCount = std::max(1, resolveConfig<int>("world_model.agentCount", 4, "FRONTEND_WORLD_AGENT_COUNT", "AI_WORLD_AGENT_COUNT"));
    const int defaultWorldMapWidth = std::max(2, resolveConfig<int>("world_model.mapWidth", 6, "FRONTEND_WORLD_MAP_WIDTH", "AI_WORLD_MAP_WIDTH"));
    const int defaultWorldMapHeight = std::max(2, resolveConfig<int>("world_model.mapHeight", 6, "FRONTEND_WORLD_MAP_HEIGHT", "AI_WORLD_MAP_HEIGHT"));
    const int defaultWorldMapDepth = std::max(1, resolveConfig<int>("world_model.mapDepth", 3, "FRONTEND_WORLD_MAP_DEPTH", "AI_WORLD_MAP_DEPTH"));
    const int defaultWorldDialogueTurns = std::max(0, resolveConfig<int>("world_model.dialogueTurns", 2, "FRONTEND_WORLD_DIALOGUE_TURNS", "AI_WORLD_DIALOGUE_TURNS"));
    const int defaultWorldEcologyClusters = std::max(0, resolveConfig<int>("world_model.ecologyClusters", 2, "FRONTEND_WORLD_ECOLOGY_CLUSTERS", "AI_WORLD_ECOLOGY_CLUSTERS"));
    const bool defaultWorld3DMap = resolveConfig<bool>("world_model.3dMapEnabled", true, "FRONTEND_WORLD_3D_MAP_ENABLED", "AI_WORLD_3D_MAP_ENABLED");
    const bool defaultWorldEmbodiedAgents = resolveConfig<bool>("world_model.embodiedAgentsEnabled", true, "FRONTEND_WORLD_EMBODIED_AGENTS_ENABLED", "AI_WORLD_EMBODIED_AGENTS_ENABLED");
    const bool defaultWorldEcologyVideo = resolveConfig<bool>("world_model.ecologyVideoEnabled", true, "FRONTEND_WORLD_ECOLOGY_VIDEO_ENABLED", "AI_WORLD_ECOLOGY_VIDEO_ENABLED");
    const bool defaultWorldPhysicsEnabled = resolveConfig<bool>("world_model.physicsEnabled", true, "FRONTEND_WORLD_PHYSICS_ENABLED", "AI_WORLD_PHYSICS_ENABLED");
    const std::string defaultWorldPhysicsBackend = resolveConfig<std::string>("world_model.physicsBackend", std::string("bullet3"), "FRONTEND_WORLD_PHYSICS_BACKEND", "AI_WORLD_PHYSICS_BACKEND");
    const int defaultWorldPhysicsSubsteps = std::max(1, resolveConfig<int>("world_model.physicsSubsteps", 4, "FRONTEND_WORLD_PHYSICS_SUBSTEPS", "AI_WORLD_PHYSICS_SUBSTEPS"));
    const bool defaultWorldEarthMapEnabled = resolveConfig<bool>("world_model.earthMapEnabled", true, "FRONTEND_WORLD_EARTH_MAP_ENABLED", "AI_WORLD_EARTH_MAP_ENABLED");
    const std::string defaultWorldEarthMapUri = resolveConfig<std::string>("world_model.earthMapUri", std::string(physics_world::bundledEarthHeightfieldUri()), "FRONTEND_WORLD_EARTH_MAP_URI", "AI_WORLD_EARTH_MAP_URI");
    const std::string defaultWorldEarthMapFormat = resolveConfig<std::string>("world_model.earthMapFormat", std::string(physics_world::preferredEarthMapFormat()), "FRONTEND_WORLD_EARTH_MAP_FORMAT", "AI_WORLD_EARTH_MAP_FORMAT");
    const fs::path bullet3Root = fs::path(resolveConfig<std::string>("frontend_server.bullet3Root", std::string("./outsides/bullet3"), "FRONTEND_BULLET3_ROOT"));
    const auto defaultEarthMapRequest = physics_world::normalizeEarthMapImportRequest(nlohmann::json{{"enabled", defaultWorldEarthMapEnabled || !defaultWorldEarthMapUri.empty()},
                                                                                                     {"sourceUri", defaultWorldEarthMapUri},
                                                                                                     {"format", defaultWorldEarthMapFormat}});
    if (!worldModelDbPath.parent_path().empty())
        fs::create_directories(worldModelDbPath.parent_path());
    if (!worldModelLegacyDir.empty())
        fs::create_directories(worldModelLegacyDir);
    static ContextService contextService(fs::path(robotsDir), useTorch);
    static std::unique_ptr<phoenix::io::JpeaV2ImageWorldModel> imageWorldModel =
        phoenix::io::createJpeaV2ImageWorldModel(
            resolveConfig<std::string>("jpea.image.variant", std::string("ijepa_vith14_1k"), "JPEA_IMAGE_VARIANT"),
            std::max(1, resolveConfig<int>("jpea.image.conceptDim", 128, "JPEA_IMAGE_CONCEPT_DIM")),
            resolveConfig<std::string>("jpea.image.backend", std::string("auto"), "JPEA_IMAGE_BACKEND"));
    static SpeakIO speakIO;
    static V51RuntimeEngine v51Runtime;
    static std::shared_ptr<Database079> worldModelDb = [worldModelDbPath, worldModelLegacyDir, worldModelRedisUrl, worldModelRedisDb, worldModelRedisPrefix]()
    {
        auto db = std::make_shared<Database079>(worldModelDbPath, worldModelLegacyDir, worldModelRedisUrl, worldModelRedisDb, worldModelRedisPrefix);
        db->open();
        return db;
    }();
    static world_model::WorldModelStore worldModel(
        worldModelDb ? worldModelDb->createStore("kvm") : nullptr,
        worldModelDb ? worldModelDb->createStore("meme_graph") : nullptr,
        worldModelDb ? worldModelDb->createStore("session") : nullptr);
    static UserStore userStore(fs::path(resolveConfig<std::string>("auth.userDb", std::string("./auth/users.json"), "AUTH_DB")));
    static phoenix::emotion::EmotionSystem emotionSystem = []() {
        phoenix::emotion::EmotionSystem::Config cfg;
        cfg.enabled = phoenix::cfgOr<bool>("emotion.enabled", true);
        cfg.storageBackend = "sqlite";
        cfg.storagePath = "./runtime_store/emotion_states.db";
        cfg.emotionDecayRate = phoenix::cfgOr<float>("emotion.decay", 0.95f);
        cfg.emotionInfluence = 0.3f;
        cfg.enableRuntimeFineTuning = true;
        cfg.historyLength = 10;
        auto &vcfg = cfg.vocabTableConfig;
        vcfg.enabled = cfg.enabled;
        vcfg.vocabTablePath = phoenix::cfgOr<std::string>("emotion.vocabTablePath", "./runtime_store/emotion_vocab_weight_table.json");
        vcfg.vocabCachePath = phoenix::cfgOr<std::string>("emotion.vocabCachePath", "./runtime_store/emotion_vocab_cache.json");
        vcfg.learningRate = phoenix::cfgOr<float>("emotion.learningRate", 0.05f);
        vcfg.maxBias = phoenix::cfgOr<float>("emotion.maxBias", 2.0f);
        vcfg.minBias = phoenix::cfgOr<float>("emotion.minBias", -2.0f);
        vcfg.decay = phoenix::cfgOr<float>("emotion.decay", 0.95f);
        vcfg.momentum = phoenix::cfgOr<float>("emotion.momentum", 0.9f);
        vcfg.tokenBoostExponent = phoenix::cfgOr<float>("emotion.tokenBoostExponent", 1.2f);
        vcfg.minTokenScore = phoenix::cfgOr<float>("emotion.minTokenScore", 0.05f);
        vcfg.applyToPrompt = phoenix::cfgOr<bool>("emotion.applyToPrompt", true);
        vcfg.applyLogitBias = phoenix::cfgOr<bool>("emotion.applyLogitBias", true);
        vcfg.stageConfigs[0].influence = phoenix::cfgOr<float>("emotion.stageWeights.immediate", 1.0f);
        vcfg.stageConfigs[0].halfLifeTurns = phoenix::cfgOr<int>("emotion.stageHalfLifeTurns.immediate", 1);
        vcfg.stageConfigs[1].influence = phoenix::cfgOr<float>("emotion.stageWeights.short", 0.7f);
        vcfg.stageConfigs[1].halfLifeTurns = phoenix::cfgOr<int>("emotion.stageHalfLifeTurns.short", 5);
        vcfg.stageConfigs[2].influence = phoenix::cfgOr<float>("emotion.stageWeights.context", 0.4f);
        vcfg.stageConfigs[2].halfLifeTurns = phoenix::cfgOr<int>("emotion.stageHalfLifeTurns.context", 20);
        vcfg.stageConfigs[3].influence = phoenix::cfgOr<float>("emotion.stageWeights.long", 0.2f);
        vcfg.stageConfigs[3].halfLifeTurns = phoenix::cfgOr<int>("emotion.stageHalfLifeTurns.long", 200);
        try {
            vcfg.seedLexicon = phoenix::cfg<std::unordered_map<std::string, std::vector<std::string>>>("emotion.seedLexicon");
        } catch (...) {
            vcfg.seedLexicon = {
                {"positive", {"happy", "great", "love", "wonderful", "glad", "excellent"}},
                {"negative", {"sad", "bad", "hate", "terrible", "angry", "awful"}},
                {"fear", {"afraid", "scared", "fear", "worried", "anxious"}},
                {"trust", {"trust", "believe", "confident", "reliable"}}
            };
        }
        return phoenix::emotion::EmotionSystem(cfg);
    }();
    static mechanical_mind::Filter mechanicalMindFilter = []() {
        mechanical_mind::Filter f;
        try { f.setEnabled(phoenix::cfgOr<bool>("mechanical_mind.enabled", false)); } catch (...) {}
        try { f.setTextThreshold(phoenix::cfgOr<double>("mechanical_mind.textThreshold", 0.58)); } catch (...) {}
        try { f.setTokenThreshold(phoenix::cfgOr<double>("mechanical_mind.tokenThreshold", 0.60)); } catch (...) {}
        try {
            std::string ph = phoenix::cfgOr<std::string>("mechanical_mind.placeholder", "[mechanized]");
            f.setPlaceholder(ph);
        } catch (...) {}
        try { f.setEmotionAware(phoenix::cfgOr<bool>("mechanical_mind.emotionAware", true)); } catch (...) {}
        try { f.setEmotionInfluence(phoenix::cfgOr<double>("mechanical_mind.emotionInfluence", 0.30)); } catch (...) {}
        try { f.setPositiveRelaxes(phoenix::cfgOr<bool>("mechanical_mind.positiveRelaxes", true)); } catch (...) {}
        return f;
    }();
    const bool allowRegister = resolveConfig<bool>("auth.allowRegister", true, "AUTH_ALLOW_REGISTER");
    const bool requireEmailVerify = resolveConfig<bool>("auth.requireEmailVerify", true, "AUTH_REQUIRE_EMAIL_VERIFY");
    const bool devReturnToken = resolveConfig<bool>("auth.devReturnToken", false, "AUTH_DEV_RETURN_TOKEN");
    const bool smtpEnabled = resolveConfig<bool>("auth.smtpEnabled", false, "AUTH_SMTP_ENABLED");
    const bool allowLocalAuthFallback = resolveConfig<bool>("auth.allowLocalTokenFallback", true, "AUTH_LOCAL_TOKEN_FALLBACK");
#ifdef _WIN32
    const bool defaultPreferLocalAuthToken = resolveConfig<bool>("auth.preferLocalToken", true);
#else
    const bool defaultPreferLocalAuthToken = resolveConfig<bool>("auth.preferLocalToken", false);
#endif
    const bool preferLocalAuthToken = resolveConfig<bool>("auth.preferLocalToken", defaultPreferLocalAuthToken, "AUTH_PREFER_LOCAL_TOKEN");
    const bool enableEmailVerifyFlow = requireEmailVerify && (smtpEnabled || devReturnToken);
    const std::string jwtSecretRuntime = resolveConfig<std::string>("auth.jwtSecret", jwtSecret, "JWT_SECRET");
    const fs::path outboxDir = fs::path(resolveConfig<std::string>("auth.outboxDir", std::string("./auth/outbox"), "AUTH_OUTBOX_DIR"));
    const std::string smtpHost = resolveConfig<std::string>("smtp.host", std::string(""), "SMTP_HOST");
    const std::string smtpPort = resolveConfig<std::string>("smtp.port", std::string("465"), "SMTP_PORT");
    const std::string smtpUser = resolveConfig<std::string>("smtp.user", std::string(""), "SMTP_USER");
    const std::string smtpPass = resolveConfig<std::string>("smtp.pass", std::string(""), "SMTP_PASS");
    const std::string smtpFrom = resolveConfig<std::string>("smtp.from", smtpUser, "SMTP_FROM");
    std::string aiApiBase = resolveConfig<std::string>("chat.aiApiBase", std::string("http://127.0.0.1:5080"), "AI_API_BASE");
    while (!aiApiBase.empty() && aiApiBase.back() == '/')
        aiApiBase.pop_back();
    auto parseThreadCount = [](const std::string &raw, int fallback)
    {
        try
        {
            int parsed = std::stoi(raw);
            return parsed;
        }
        catch (...)
        {
            return fallback;
        }
    };
    unsigned hw = std::thread::hardware_concurrency();
    int defaultThreads = hw > 0 ? static_cast<int>(std::max(2u, std::min(8u, hw))) : 4;
    int frontendThreads = std::max(2, resolveConfig<int>("frontend_server.httpThreads", defaultThreads, "FRONTEND_HTTP_THREADS"));
    if (frontendThreads < 2)
        frontendThreads = 2;
    drogon::app().setThreadNum(frontendThreads);

    auto issueToken = [jwtSecretRuntime, allowLocalAuthFallback, preferLocalAuthToken](const UserRecord &user)
    {
        if (allowLocalAuthFallback && preferLocalAuthToken)
            return issueLocalAuthToken(user.username, 7LL * 24LL * 60LL * 60LL * 1000LL);
        try
        {
            return jwt::create()
                .set_type("JWT")
                .set_subject(user.username)
                .set_payload_claim("role", jwt::claim(user.role))
                .sign(jwt::algorithm::hs256{jwtSecretRuntime});
        }
        catch (...)
        {
            if (!allowLocalAuthFallback)
                throw;
            return issueLocalAuthToken(user.username, 7LL * 24LL * 60LL * 60LL * 1000LL);
        }
    };

    auto parseToken = [jwtSecretRuntime, allowLocalAuthFallback](const drogon::HttpRequestPtr &req, UserRecord &out) -> bool
    {
        std::string token = extractBearerToken(req);
        if (token.empty())
            return false;
        if (allowLocalAuthFallback && token.rfind("local-", 0) == 0)
        {
            std::string username;
            if (parseLocalAuthToken(token, username))
            {
                out.username = username;
                return true;
            }
            return false;
        }
        try
        {
            auto decoded = jwt::decode(token);
            auto verifier = jwt::verify()
                                .allow_algorithm(jwt::algorithm::hs256{jwtSecretRuntime});
            verifier.verify(decoded);
            out.username = decoded.get_subject();
            if (decoded.has_payload_claim("role"))
                out.role = decoded.get_payload_claim("role").as_string();
            return true;
        }
        catch (...)
        {
            if (!allowLocalAuthFallback)
                return false;
            std::string username;
            if (parseLocalAuthToken(token, username))
            {
                out.username = username;
                return true;
            }
            return false;
        }
    };

    auto resolveAuthenticatedUser = [parseToken, &userStore](const drogon::HttpRequestPtr &req, UserRecord &full) -> bool
    {
        UserRecord parsed;
        if (!parseToken(req, parsed))
            return false;
        return userStore.getUser(parsed.username, full);
    };

    auto deliverToken = [&outboxDir, devReturnToken, smtpEnabled, smtpHost, smtpPort, smtpUser, smtpPass, smtpFrom](
                            const std::string &email, const std::string &kind, const std::string &token)
    {
#ifdef HAVE_CURL
        if (smtpEnabled && !smtpHost.empty() && !smtpUser.empty() && !smtpPass.empty())
        {
            CURL *curl = curl_easy_init();
            if (curl)
            {
                struct curl_slist *recipients = nullptr;
                std::ostringstream url;
                url << "smtps://" << smtpHost << ":" << smtpPort;
                std::ostringstream payload;
                payload << "To: " << email << "\r\n";
                payload << "From: " << smtpFrom << "\r\n";
                payload << "Subject: " << (kind == "verify" ? "Verify your email" : "Reset your password") << "\r\n";
                payload << "\r\n";
                payload << "Token: " << token << "\r\n";
                std::string data = payload.str();
                size_t pos = 0;
                auto readCb = [](char *ptr, size_t size, size_t nmemb, void *userdata) -> size_t
                {
                    auto *ctx = reinterpret_cast<std::pair<std::string *, size_t *> *>(userdata);
                    size_t max = size * nmemb;
                    if (*ctx->second >= ctx->first->size())
                        return 0;
                    size_t remain = ctx->first->size() - *ctx->second;
                    size_t n = std::min(max, remain);
                    std::memcpy(ptr, ctx->first->data() + *ctx->second, n);
                    *ctx->second += n;
                    return n;
                };
                std::pair<std::string *, size_t *> ctx{&data, &pos};
                curl_easy_setopt(curl, CURLOPT_URL, url.str().c_str());
                curl_easy_setopt(curl, CURLOPT_USERNAME, smtpUser.c_str());
                curl_easy_setopt(curl, CURLOPT_PASSWORD, smtpPass.c_str());
                curl_easy_setopt(curl, CURLOPT_USE_SSL, (long)CURLUSESSL_ALL);
                curl_easy_setopt(curl, CURLOPT_MAIL_FROM, smtpFrom.c_str());
                recipients = curl_slist_append(recipients, email.c_str());
                curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);
                curl_easy_setopt(curl, CURLOPT_READFUNCTION, readCb);
                curl_easy_setopt(curl, CURLOPT_READDATA, &ctx);
                curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
                CURLcode res = curl_easy_perform(curl);
                curl_slist_free_all(recipients);
                curl_easy_cleanup(curl);
                if (res == CURLE_OK)
                {
                    return devReturnToken ? token : std::string();
                }
            }
        }
#endif
        fs::create_directories(outboxDir);
        std::ostringstream name;
        name << email << "_" << kind << "_" << nowEpochMs() << ".txt";
        fs::path file = outboxDir / name.str();
        std::ofstream out(file);
        out << "To: " << email << "\n";
        out << "Type: " << kind << "\n";
        out << "Token: " << token << "\n";
        out << "GeneratedAt: " << nowEpochMs() << "\n";
        return devReturnToken ? token : std::string();
    };

    auto parseIntEnv = [](const std::string &value, int fallback)
    {
        try
        {
            int parsed = std::stoi(value);
            return parsed;
        }
        catch (...)
        {
            return fallback;
        }
    };
    const int chatQueueWaitMs = std::max(3000, resolveConfig<int>("chat.queueWaitMs", 180000, "FRONTEND_CHAT_QUEUE_WAIT_MS"));
    const int chatUpstreamTimeoutMs = std::max(5000, resolveConfig<int>("chat.upstreamTimeoutMs", 360000, "FRONTEND_CHAT_UPSTREAM_TIMEOUT_MS"));
    const int apiUpstreamTimeoutMs = std::max(3000, resolveConfig<int>("api.upstreamTimeoutMs", 45000, "FRONTEND_API_UPSTREAM_TIMEOUT_MS"));
    const int chatMaxInFlight = std::max(1, std::min(16, resolveConfig<int>("chat.maxInFlight", 1, "FRONTEND_CHAT_MAX_INFLIGHT")));
    const bool frontendHttpLog = resolveConfig<bool>("frontend_server.httpLog", false, "FRONTEND_HTTP_LOG");
    const bool frontendV51Default = resolveConfig<bool>("frontend_server.v51Default", true, "FRONTEND_V51_DEFAULT");
    static edge_platform::PlatformManager platformManager;
    static std::once_flag platformInitOnce;

    auto splitPathList = [](const std::string &raw) {
        std::vector<fs::path> paths;
        std::string token;
        for (char ch : raw) {
            if (ch == ';' || ch == ',' || ch == '\n' || ch == '\r') {
                if (!token.empty()) {
                    paths.emplace_back(token);
                    token.clear();
                }
                continue;
            }
            token.push_back(ch);
        }
        if (!token.empty()) {
            paths.emplace_back(token);
        }
        return paths;
    };

    std::call_once(platformInitOnce, [&]() {
        edge_platform::RuntimeConfig platformConfig;
        platformConfig.baseDir = fs::path(resolveConfig<std::string>("edge_platform.baseDir", fs::current_path().string(), "FRONTEND_PLATFORM_BASE_DIR", "EDGE_PLATFORM_BASE_DIR"));
        platformConfig.netlistRoot = fs::path(resolveConfig<std::string>("edge_platform.netlistRoot", std::string("./catastrophe"), "FRONTEND_PLATFORM_NETLIST_ROOT", "EDGE_PLATFORM_NETLIST_ROOT"));
        platformConfig.gerberConnectorMap = fs::path(resolveConfig<std::string>("edge_platform.gerberConnectorMap", std::string("./catastrophe/gerber_catastrophe1_20260417_connector_map.json"), "FRONTEND_PLATFORM_GERBER_CONNECTOR_MAP", "EDGE_PLATFORM_GERBER_CONNECTOR_MAP"));
        platformConfig.enabled = resolveConfig<bool>("edge_platform.enabled", true, "FRONTEND_PLATFORM_ENABLED", "EDGE_PLATFORM_ENABLED");
        platformConfig.npuEnabled = resolveConfig<bool>("edge_platform.npu.enabled", true, "FRONTEND_PLATFORM_NPU_ENABLED", "EDGE_PLATFORM_NPU_ENABLED");
        platformConfig.preferredComputeBackend = resolveConfig<std::string>("edge_platform.preferredComputeBackend", std::string("auto"), "FRONTEND_PLATFORM_PREFERRED_BACKEND", "EDGE_PLATFORM_PREFERRED_BACKEND");
        platformConfig.maxComputeInflight = std::max(1, resolveConfig<int>("edge_platform.maxComputeInflight", 2, "FRONTEND_PLATFORM_MAX_COMPUTE_INFLIGHT", "EDGE_PLATFORM_MAX_COMPUTE_INFLIGHT"));
        platformConfig.npuSpiDevice = resolveConfig<std::string>("edge_platform.npu.spiDevice", std::string("/dev/spidev0.0"), "FRONTEND_PLATFORM_SPI_DEVICE", "EDGE_PLATFORM_SPI_DEVICE");
        platformConfig.npuSpiSpeedHz = std::max(100000, resolveConfig<int>("edge_platform.npu.spiSpeedHz", 120000000, "FRONTEND_PLATFORM_SPI_SPEED_HZ", "EDGE_PLATFORM_SPI_SPEED_HZ"));
        platformConfig.npuSpiMode = std::max(0, std::min(3, resolveConfig<int>("edge_platform.npu.spiMode", 0, "FRONTEND_PLATFORM_SPI_MODE", "EDGE_PLATFORM_SPI_MODE")));
        platformConfig.npuAsyncGpioExecuteLocally = resolveConfig<bool>("edge_platform.npu.gpioLocalExecute", true, "FRONTEND_PLATFORM_GPIO_LOCAL_EXECUTE", "EDGE_PLATFORM_GPIO_LOCAL_EXECUTE");
        platformConfig.npuGpioSysfsRoot = resolveConfig<std::string>("edge_platform.npu.gpioSysfsRoot", std::string("/sys/class/gpio"), "FRONTEND_PLATFORM_GPIO_SYSFS_ROOT", "EDGE_PLATFORM_GPIO_SYSFS_ROOT");
        platformConfig.npuGpioPulseUs = std::max(0, resolveConfig<int>("edge_platform.npu.gpioPulseUs", 0, "FRONTEND_PLATFORM_GPIO_PULSE_US", "EDGE_PLATFORM_GPIO_PULSE_US"));
        platformConfig.npuVirtualMemoryEnabled = resolveConfig<bool>("edge_platform.npu.vmEnabled", true, "FRONTEND_PLATFORM_NPU_VM_ENABLED", "EDGE_PLATFORM_NPU_VM_ENABLED");
        platformConfig.npuAdvertisedMemoryMb = std::max(1024, resolveConfig<int>("edge_platform.npu.vmMb", 8192, "FRONTEND_PLATFORM_NPU_VM_MB", "EDGE_PLATFORM_NPU_VM_MB"));
        platformConfig.npuSdcardWeightsRoot = fs::path(resolveConfig<std::string>("edge_platform.npu.sdWeightsRoot", std::string("runtime_store/sdcard_weights"), "FRONTEND_PLATFORM_NPU_SD_WEIGHTS_ROOT", "EDGE_PLATFORM_NPU_SD_WEIGHTS_ROOT"));
        platformConfig.npuHotWeightsLimit = std::max(1, resolveConfig<int>("edge_platform.npu.hotWeightsLimit", 12, "FRONTEND_PLATFORM_NPU_HOT_LIMIT", "EDGE_PLATFORM_NPU_HOT_LIMIT"));
        platformConfig.npuHotPromoteHits = std::max(1, resolveConfig<int>("edge_platform.npu.hotPromoteHits", 2, "FRONTEND_PLATFORM_NPU_HOT_PROMOTE_HITS", "EDGE_PLATFORM_NPU_HOT_PROMOTE_HITS"));
        platformConfig.npuUnitCount = std::max(1, resolveConfig<int>("edge_platform.npu.unitCount", 19, "FRONTEND_PLATFORM_NPU_UNIT_COUNT", "EDGE_PLATFORM_NPU_UNIT_COUNT"));
        platformConfig.npuEfficiencyProbeEnabled = resolveConfig<bool>("edge_platform.npu.probeEnabled", true, "FRONTEND_PLATFORM_NPU_PROBE_ENABLED", "EDGE_PLATFORM_NPU_PROBE_ENABLED");
        platformConfig.npuEfficiencyAnomalyThreshold = std::max(0.05, std::min(1.0, (double)resolveConfig<float>("edge_platform.npu.probeThreshold", 0.35f, "FRONTEND_PLATFORM_NPU_PROBE_THRESHOLD", "EDGE_PLATFORM_NPU_PROBE_THRESHOLD")));
        const auto netlistFiles = splitPathList(resolveConfig<std::string>("edge_platform.netlistFiles", std::string(), "FRONTEND_PLATFORM_NETLIST_FILES", "EDGE_PLATFORM_NETLIST_FILES"));
        if (!netlistFiles.empty()) {
            platformConfig.netlistFiles = netlistFiles;
        }
        platformManager.reconfigure(platformConfig);
        platformManager.refreshTopology();
    });

    auto shouldRunV51 = [frontendV51Default](const Json::Value *json) -> bool
    {
        if (!json)
            return frontendV51Default;
        if (json->isMember("useV51"))
            return (*json)["useV51"].asBool();
        return frontendV51Default;
    };

    auto buildV51Request = [](const Json::Value *json, const std::string &sessionId, const std::string &text,
                              const std::string &imageCtx = std::string(), const std::string &speechCtx = std::string())
    {
        Json::Value req;
        req["sessionId"] = sessionId;
        req["text"] = text;
        if (!imageCtx.empty())
            req["imageContext"] = imageCtx;
        if (!speechCtx.empty())
            req["speechContext"] = speechCtx;
        if (json)
        {
            if (json->isMember("domain") && (*json)["domain"].isString())
                req["domain"] = (*json)["domain"].asString();
            if (json->isMember("domainHints") && (*json)["domainHints"].isArray())
                req["domainHints"] = (*json)["domainHints"];
            if (json->isMember("imageContext") && (*json)["imageContext"].isString())
                req["imageContext"] = (*json)["imageContext"].asString();
            if (json->isMember("speechContext") && (*json)["speechContext"].isString())
                req["speechContext"] = (*json)["speechContext"].asString();
            if (json->isMember("videoContext") && (*json)["videoContext"].isString())
                req["videoContext"] = (*json)["videoContext"].asString();
        }
        return req;
    };

    auto proxyApiCall = [aiApiBase, chatQueueWaitMs, chatUpstreamTimeoutMs, apiUpstreamTimeoutMs, chatMaxInFlight, frontendHttpLog, &contextService](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
    {
        std::cout << "[proxyApiCall] ENTER, path=" << req->path() << ", method=" << req->method() << std::endl;
        bool chatPath = false;
        // 聊天会话 ID：用于异步回调把助手回复写回 ContextService（实现 session 内记忆）。
        auto chatSessionId = std::make_shared<std::string>();
        auto dispatchStarted = std::chrono::steady_clock::now();
        auto callback = std::make_shared<std::function<void(const drogon::HttpResponsePtr &)>>(std::move(cb));
        auto callbackDone = std::make_shared<std::atomic<bool>>(false);
        auto finishOnce = [callback, callbackDone](const drogon::HttpResponsePtr &resp) -> bool
        {
            bool expected = false;
            if (callbackDone->compare_exchange_strong(expected, true))
            {
                (*callback)(resp);
                return true;
            }
            return false;
        };
        auto chatSlotHeld = std::make_shared<std::atomic<bool>>(false);
        auto releaseChatSlot = [chatSlotHeld]()
        {
            bool expected = true;
            if (chatSlotHeld->compare_exchange_strong(expected, false))
            {
                int prev = gChatProxyInFlight.fetch_sub(1);
                if (prev <= 0)
                    gChatProxyInFlight.store(0);
            }
        };
        try
        {
            std::cout << "[proxyApiCall] In try block" << std::endl;
            auto apiClient = drogon::HttpClient::newHttpClient(aiApiBase);
            auto isChatPath = [](const std::string &path)
            {
                return path == "/api/chat" || path == "/api/transformer/chat";
            };

        auto outgoing = drogon::HttpRequest::newHttpRequest();
        outgoing->setMethod(req->method());

        std::string routePath = req->path();
        std::cout << "[proxyApiCall] routePath=" << routePath << std::endl;
        if (routePath.size() > 1 && routePath.back() == '/')
        {
            bool apiPath = routePath.rfind("/api/", 0) == 0;
            bool robotsPath = routePath.rfind("/robots/", 0) == 0;
            if (apiPath || robotsPath)
                routePath.pop_back();
        }
        std::string path = routePath;
        if (!req->query().empty())
            path += "?" + req->query();
        outgoing->setPath(path);

        std::string requestContentType = req->getHeader("content-type");

        chatPath = isChatPath(routePath);
        std::cout << "[proxyApiCall] chatPath=" << chatPath << std::endl;
        int upstreamTimeoutMs = chatPath ? chatUpstreamTimeoutMs : apiUpstreamTimeoutMs;
        if (!req->body().empty())
        {
            if (chatPath)
            {
                outgoing->setContentTypeCode(drogon::CT_APPLICATION_JSON);
                // ── 上下文路由集成：把 ContextService 接入主聊天流 ──────────────
                // 解析聊天请求，按 sessionId 构建会话历史（concat/rnn/lstm 路由），
                // 在前端未显式提供 contextHint 时自动注入，实现 session 内记忆。
                auto chatJson = req->getJsonObject();
                bool injected = false;
                if (chatJson && chatJson->isMember("text") && (*chatJson)["text"].isString())
                {
                    std::string sid;
                    if (chatJson->isMember("sessionId") && (*chatJson)["sessionId"].isString())
                        sid = (*chatJson)["sessionId"].asString();
                    const std::string userText = (*chatJson)["text"].asString();
                    std::string modeHint = "auto";
                    if (chatJson->isMember("contextMode") && (*chatJson)["contextMode"].isString())
                        modeHint = (*chatJson)["contextMode"].asString();
                    std::cout << "[frontend-proxy] chatJson exists, sid=" << sid << ", userText=" << userText << ", modeHint=" << modeHint << std::endl;
                    if (!sid.empty() && !userText.empty())
                    {
                        try
                        {
                            std::string routedHint = contextService.prepareChatContext(sid, userText, modeHint);
                            bool frontendHint = chatJson->isMember("contextHint") && (*chatJson)["contextHint"].isString() && !(*chatJson)["contextHint"].asString().empty();
                            std::cout << "[frontend-proxy] prepareChatContext: sid=" << sid << ", routedHint=" << (routedHint.empty() ? "(empty)" : routedHint.substr(0, 100)) << ", frontendHint=" << frontendHint << std::endl;
                            if (!frontendHint)
                            {
                                // Always set the key (even empty) so the main gateway can detect
                                // that the frontend already handled context preparation and does not
                                // call /context/hint again.
                                (*chatJson)["contextHint"] = routedHint;
                                if (!chatJson->isMember("contextWeight") || !(*chatJson)["contextWeight"].isNumeric())
                                    (*chatJson)["contextWeight"] = 0.9;
                                std::cout << "[frontend-proxy] Set contextHint weight=0.9" << std::endl;
                            }
                            *chatSessionId = sid; // 供异步回调写回助手回复
                            outgoing->setBody(chatJson->toStyledString());
                            injected = true;
                        }
                        catch (...)
                        {
                        }
                    }
                    else
                    {
                        std::cout << "[frontend-proxy] sid or userText empty, skipping prepareChatContext" << std::endl;
                    }
                }
                else
                {
                    std::cout << "[frontend-proxy] chatJson invalid or missing text field" << std::endl;
                }
                if (!injected)
                    outgoing->setBody(std::string(req->body()));
            }
            else
            {
                std::string lowerCt = requestContentType;
                std::transform(lowerCt.begin(), lowerCt.end(), lowerCt.begin(), [](unsigned char c)
                               { return static_cast<char>(std::tolower(c)); });
                if (lowerCt.find("application/json") != std::string::npos)
                {
                    outgoing->setContentTypeCode(drogon::CT_APPLICATION_JSON);
                    auto json = req->getJsonObject();
                    if (json)
                    {
                        outgoing->setBody(json->toStyledString());
                    }
                    else
                    {
                        outgoing->setBody(std::string(req->body()));
                    }
                }
                else
                {
                    outgoing->setBody(std::string(req->body()));
                    if (!requestContentType.empty())
                        outgoing->addHeader("content-type", requestContentType);
                }
            }
        }

        if (chatPath)
        {
            auto started = std::chrono::steady_clock::now();
            bool acquired = false;
            long long queueWaitMs = 0;
            while (!acquired)
            {
                int current = gChatProxyInFlight.load();
                while (current < chatMaxInFlight)
                {
                    if (gChatProxyInFlight.compare_exchange_weak(current, current + 1))
                    {
                        acquired = true;
                        chatSlotHeld->store(true);
                        queueWaitMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
                        break;
                    }
                }
                if (acquired)
                {
                    break;
                }
                auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
                if (waited > chatQueueWaitMs)
                {
                    auto resp = drogon::HttpResponse::newHttpResponse();
                    resp->setStatusCode(drogon::k200OK);
                    resp->setContentTypeString("application/json");
                    Json::Value out;
                    out["ok"] = false;
                    out["error"] = "request-timeout:/api/chat";
                    out["connected"] = false;
                    out["stage"] = "frontend-queue";
                    out["waitMs"] = (Json::Int64)waited;
                    out["queueTimeoutMs"] = chatQueueWaitMs;
                    out["maxInFlight"] = chatMaxInFlight;
                    out["inFlight"] = gChatProxyInFlight.load();
                    resp->setBody(out.toStyledString());
                    if (frontendHttpLog)
                    {
                        std::cout << "[frontend-proxy] chat queue timeout path=" << routePath
                                  << " waitMs=" << waited
                                  << " queueTimeoutMs=" << chatQueueWaitMs
                                  << " inFlight=" << gChatProxyInFlight.load() << "/" << chatMaxInFlight << std::endl;
                    }
                    (void)finishOnce(resp);
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }

            {
                std::lock_guard<std::mutex> lock(gChatDispatchMu);
                auto now = std::chrono::steady_clock::now();
                auto since = std::chrono::duration_cast<std::chrono::milliseconds>(now - gLastChatDispatch);
                constexpr auto minGap = std::chrono::milliseconds(220);
                if (since < minGap)
                    std::this_thread::sleep_for(minGap - since);
                gLastChatDispatch = std::chrono::steady_clock::now();
            }

            auto auth = req->getHeader("authorization");
            if (!auth.empty())
                outgoing->addHeader("authorization", auth);
            if (frontendHttpLog)
            {
                std::cout << "[frontend-proxy] chat dispatch path=" << routePath
                          << " queueWaitMs=" << queueWaitMs
                          << " upstreamTimeoutMs=" << upstreamTimeoutMs
                          << " inFlight=" << gChatProxyInFlight.load() << "/" << chatMaxInFlight << std::endl;
            }
        }
        else
        {
            for (const auto &header : req->headers())
            {
                std::string keyLower = header.first;
                std::transform(keyLower.begin(), keyLower.end(), keyLower.begin(), [](unsigned char c)
                               { return static_cast<char>(std::tolower(c)); });
                if (keyLower == "host" || keyLower == "connection" || keyLower == "content-length" || keyLower == "content-type")
                    continue;
                outgoing->addHeader(header.first, header.second);
            }
        }

            apiClient->sendRequest(
                outgoing,
                [finishOnce, releaseChatSlot, apiClient, chatPath, routePath, dispatchStarted, frontendHttpLog, upstreamTimeoutMs, &contextService, chatSessionId](drogon::ReqResult result, const drogon::HttpResponsePtr &upstreamResp) mutable
                {
                    try
                    {
                        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - dispatchStarted).count();
                        if (result != drogon::ReqResult::Ok || !upstreamResp)
                        {
                            releaseChatSlot();
                            auto resp = drogon::HttpResponse::newHttpResponse();
                            resp->setStatusCode(chatPath ? drogon::k200OK : drogon::k503ServiceUnavailable);
                            resp->setContentTypeString("application/json");
                            Json::Value out;
                            if (chatPath)
                            {
                                out["ok"] = false;
                                out["error"] = "request-timeout:/api/chat";
                                out["connected"] = false;
                                out["stage"] = "frontend-upstream";
                                out["elapsedMs"] = (Json::Int64)elapsedMs;
                                out["upstreamTimeoutMs"] = upstreamTimeoutMs;
                                out["reqResult"] = static_cast<int>(result);
                            }
                            else
                            {
                                out["ok"] = false;
                                out["error"] = "disconnected";
                            }
                            resp->setBody(out.toStyledString());
                            if (frontendHttpLog)
                            {
                                std::cout << "[frontend-proxy] upstream fail path=" << routePath
                                          << " result=" << static_cast<int>(result)
                                          << " elapsedMs=" << elapsedMs
                                          << " timeoutMs=" << upstreamTimeoutMs
                                          << " chat=" << (chatPath ? "true" : "false") << std::endl;
                            }
                            (void)finishOnce(resp);
                            return;
                        }

                        auto resp = drogon::HttpResponse::newHttpResponse();
                        resp->setStatusCode(upstreamResp->statusCode());
                        resp->setBody(std::string(upstreamResp->body()));
                        releaseChatSlot();

                        auto upstreamContentType = upstreamResp->getHeader("content-type");
                        if (!upstreamContentType.empty())
                        {
                            resp->setContentTypeString(upstreamContentType);
                        }

                        for (const auto &header : upstreamResp->headers())
                        {
                            std::string keyLower = header.first;
                            std::transform(keyLower.begin(), keyLower.end(), keyLower.begin(), [](unsigned char c)
                                           { return static_cast<char>(std::tolower(c)); });
                            if (keyLower == "connection" || keyLower == "transfer-encoding" || keyLower == "content-length" || keyLower == "content-type" || keyLower == "server" || keyLower == "date")
                                continue;
                            resp->addHeader(header.first, header.second);
                        }

                        if (frontendHttpLog && chatPath)
                        {
                            std::cout << "[frontend-proxy] chat upstream ok path=" << routePath
                                      << " status=" << (int)upstreamResp->statusCode()
                                      << " elapsedMs=" << elapsedMs << std::endl;
                        }

                        // ── 记忆回写：把助手回复写回 ContextService 会话窗口 ──────────
                        // 使下一轮 prepareChatContext 能渲染出本轮的助手回答，
                        // 实现真正的 session 内多轮记忆。
                        if (chatPath && chatSessionId && !chatSessionId->empty())
                        {
                            try
                            {
                                auto upstreamJson = upstreamResp->getJsonObject();
                                if (upstreamJson)
                                {
                                    const Json::Value &rj = *upstreamJson;
                                    std::string reply;
                                    if (rj.isMember("result") && rj["result"].isObject())
                                    {
                                        const Json::Value &r = rj["result"];
                                        if (r.isMember("reply") && r["reply"].isString())
                                            reply = r["reply"].asString();
                                        else if (r.isMember("message") && r["message"].isString())
                                            reply = r["message"].asString();
                                    }
                                    if (reply.empty() && rj.isMember("reply") && rj["reply"].isString())
                                        reply = rj["reply"].asString();
                                    if (!reply.empty() && reply != "disconnected")
                                        contextService.recordAssistantReply(*chatSessionId, reply);
                                }
                            }
                            catch (...)
                            {
                            }
                        }

                        (void)finishOnce(resp);
                    }
                    catch (...)
                    {
                        releaseChatSlot();
                        auto resp = drogon::HttpResponse::newHttpResponse();
                        resp->setStatusCode(drogon::k500InternalServerError);
                        resp->setContentTypeString("application/json");
                        Json::Value out;
                        out["ok"] = false;
                        out["error"] = "proxy-callback-exception";
                        resp->setBody(out.toStyledString());
                        (void)finishOnce(resp);
                    }
                });

            std::thread([finishOnce, releaseChatSlot, chatPath, routePath, upstreamTimeoutMs, frontendHttpLog]()
                        {
                std::this_thread::sleep_for(std::chrono::milliseconds(upstreamTimeoutMs));
                releaseChatSlot();
                auto resp = drogon::HttpResponse::newHttpResponse();
                resp->setStatusCode(chatPath ? drogon::k200OK : drogon::k504GatewayTimeout);
                resp->setContentTypeString("application/json");
                Json::Value out;
                out["ok"] = false;
                out["error"] = chatPath ? "request-timeout:/api/chat" : "request-timeout";
                out["stage"] = "frontend-watchdog";
                out["timeoutMs"] = upstreamTimeoutMs;
                resp->setBody(out.toStyledString());
                bool fired = finishOnce(resp);
                if (frontendHttpLog && fired)
                {
                    std::cout << "[frontend-proxy] watchdog timeout path=" << routePath
                              << " timeoutMs=" << upstreamTimeoutMs
                              << " chat=" << (chatPath ? "true" : "false") << std::endl;
                } })
                .detach();
        }
        catch (...)
        {
            std::cout << "[proxyApiCall] Exception caught in proxyApiCall" << std::endl;
            releaseChatSlot();
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            Json::Value out;
            out["ok"] = false;
            out["error"] = "proxy-dispatch-exception";
            resp->setBody(out.toStyledString());
            (void)finishOnce(resp);
        }
    };

    if (!fs::exists(webRoot))
    {
        std::cerr << "[frontend_server] WEB_ROOT does not exist: " << webRoot << std::endl;
        return;
    }

    if (frontendHttpLog)
    {
        drogon::app().registerPostHandlingAdvice([](const drogon::HttpRequestPtr &req, const drogon::HttpResponsePtr &resp)
                                                 {
            try {
                auto &logger = LoggerCXX::instance();
                int status = resp ? (int)resp->statusCode() : 0;
                std::string path = req ? req->path() : std::string();
                std::string method = req ? req->methodString() : std::string("UNKNOWN");
                std::string msg = std::string("frontend route=") + method + " " + path + " status=" + std::to_string(status);
                std::cout << "[frontend] " << method << " " << path << " -> " << status << std::endl;
                if (logger.enabled()) {
                    if (status >= 500) logger.log(LoggerCXX::Type::ERROR, msg);
                    else if (status >= 400) logger.log(LoggerCXX::Type::WARNING, msg);
                    else logger.log(LoggerCXX::Type::LOG, msg);
                }
            } catch (...) {
            } });
    }

    drogon::app().registerHandler("/", [webRoot](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        const std::string indexFile = webRoot + "/index.html";

        if (!fs::exists(indexFile)) {
            resp->setStatusCode(drogon::k404NotFound);
            resp->setBody("index.html not found");
        } else {
            std::ifstream file(indexFile);
            std::stringstream buffer;
            buffer << file.rdbuf();
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("text/html; charset=utf-8");
            resp->setBody(buffer.str());
        }
        cb(resp); });

    auto serveStaticFile = [webRoot](const fs::path &relative, const std::string &contentType, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
    {
        auto resp = drogon::HttpResponse::newHttpResponse();
        fs::path base = fs::path(webRoot);
        fs::path file = (base / relative).lexically_normal();
        auto baseStr = base.lexically_normal().string();
        auto fileStr = file.string();
        if (fileStr.size() < baseStr.size() || fileStr.rfind(baseStr, 0) != 0 || !fs::exists(file) || !fs::is_regular_file(file))
        {
            resp->setStatusCode(drogon::k404NotFound);
            resp->setBody("Not Found");
            cb(resp);
            return;
        }

        std::ifstream in(file, std::ios::binary);
        if (!in)
        {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody("Failed to read file");
            cb(resp);
            return;
        }
        std::ostringstream buffer;
        buffer << in.rdbuf();
        resp->setStatusCode(drogon::k200OK);
        if (!contentType.empty())
            resp->setContentTypeString(contentType);
        resp->setBody(buffer.str());
        cb(resp);
    };

    drogon::app().registerHandler("/static/js/{1}", [serveStaticFile](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb, const std::string &name)
                                  {
        if (name != fs::path(name).filename().string()) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k400BadRequest);
            resp->setBody("bad file name");
            cb(resp);
            return;
        }
        std::string contentType = (name.size() >= 4 && name.substr(name.size() - 4) == ".map") ? "application/json" : "application/javascript; charset=utf-8";
        serveStaticFile(fs::path("static") / "js" / name, contentType, std::move(cb)); }, {drogon::Get});

    drogon::app().registerHandler("/static/css/{1}", [serveStaticFile](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb, const std::string &name)
                                  {
        if (name != fs::path(name).filename().string()) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k400BadRequest);
            resp->setBody("bad file name");
            cb(resp);
            return;
        }
        serveStaticFile(fs::path("static") / "css" / name, "text/css; charset=utf-8", std::move(cb)); }, {drogon::Get});

    drogon::app().registerHandler("/manifest.json", [serveStaticFile](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        serveStaticFile("manifest.json", "application/json", std::move(cb)); }, {drogon::Get});

    drogon::app().registerHandler("/asset-manifest.json", [serveStaticFile](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        serveStaticFile("asset-manifest.json", "application/json", std::move(cb)); }, {drogon::Get});

    drogon::app().registerHandler("/logo192.png", [serveStaticFile, webRoot](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        fs::path logoPath = fs::path(webRoot) / "logo192.png";
        if (fs::exists(logoPath) && fs::is_regular_file(logoPath)) {
            serveStaticFile("logo192.png", "image/png", std::move(cb));
            return;
        }
        static const unsigned char kOnePixelPng[] = {
            0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,
            0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x08,0x04,0x00,0x00,0x00,0xB5,0x1C,0x0C,
            0x02,0x00,0x00,0x00,0x0B,0x49,0x44,0x41,0x54,0x78,0xDA,0x63,0xFC,0x5F,0x0F,0x00,
            0x02,0x7F,0x01,0xF5,0x7E,0x6F,0xB8,0xA7,0x00,0x00,0x00,0x00,0x49,0x45,0x4E,0x44,
            0xAE,0x42,0x60,0x82
        };
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k200OK);
        resp->setContentTypeString("image/png");
        resp->setBody(std::string(reinterpret_cast<const char *>(kOnePixelPng), sizeof(kOnePixelPng)));
        cb(resp); }, {drogon::Get});

    drogon::app().registerHandler("/logo512.png", [serveStaticFile, webRoot](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        fs::path logoPath = fs::path(webRoot) / "logo512.png";
        if (fs::exists(logoPath) && fs::is_regular_file(logoPath)) {
            serveStaticFile("logo512.png", "image/png", std::move(cb));
            return;
        }
        static const unsigned char kOnePixelPng[] = {
            0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,
            0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x08,0x04,0x00,0x00,0x00,0xB5,0x1C,0x0C,
            0x02,0x00,0x00,0x00,0x0B,0x49,0x44,0x41,0x54,0x78,0xDA,0x63,0xFC,0x5F,0x0F,0x00,
            0x02,0x7F,0x01,0xF5,0x7E,0x6F,0xB8,0xA7,0x00,0x00,0x00,0x00,0x49,0x45,0x4E,0x44,
            0xAE,0x42,0x60,0x82
        };
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k200OK);
        resp->setContentTypeString("image/png");
        resp->setBody(std::string(reinterpret_cast<const char *>(kOnePixelPng), sizeof(kOnePixelPng)));
        cb(resp); }, {drogon::Get});

    drogon::app().registerHandler("/robots.txt", [serveStaticFile](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        serveStaticFile("robots.txt", "text/plain; charset=utf-8", std::move(cb)); }, {drogon::Get});

    drogon::app().registerHandler("/favicon.ico", [serveStaticFile](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        serveStaticFile("favicon.ico", "image/x-icon", std::move(cb)); }, {drogon::Get});

    drogon::app().registerHandler("/api/health", [](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k200OK);
        resp->setContentTypeString("application/json");
        Json::Value out;
        out["ok"] = true;
        out["service"] = "phoenix-frontend";
        out["timestamp"] = static_cast<Json::UInt64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        resp->setBody(out.toStyledString());
        cb(resp); }, {drogon::Get, drogon::Post});

    drogon::app().registerHandler("/auth/config", [&userStore, allowRegister, enableEmailVerifyFlow, allowLocalAuthFallback, preferLocalAuthToken, devReturnToken](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        Json::Value out;
        out["ok"] = true;
        out["allowBootstrap"] = !userStore.hasUsers();
        out["allowRegister"] = allowRegister;
        out["requireEmailVerify"] = enableEmailVerifyFlow;
        out["localTokenFallback"] = allowLocalAuthFallback;
        out["preferLocalToken"] = preferLocalAuthToken;
        out["devReturnToken"] = devReturnToken;
        resp->setStatusCode(drogon::k200OK);
        resp->setContentTypeString("application/json");
        resp->setBody(out.toStyledString());
        cb(resp); });

    drogon::app().registerHandler("/auth/bootstrap", [&userStore, issueToken, enableEmailVerifyFlow, deliverToken](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            if (userStore.hasUsers()) {
                sendAuthErrorJson(cb, drogon::k409Conflict, "already bootstrapped");
                return;
            }
            auto json = req->getJsonObject();
            if (!json || !json->isMember("username") || !json->isMember("password") || !json->isMember("email")) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, "Missing username, password or email");
                return;
            }
            std::string err;
            const std::string username = (*json)["username"].asString();
            const std::string password = (*json)["password"].asString();
            const std::string email = (*json)["email"].asString();
            if (!userStore.addUser(username, email, password, "admin", err)) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, err.empty() ? std::string("bootstrap failed") : err);
                return;
            }
            UserRecord rec;
            userStore.getUser(username, rec);
            Json::Value out;
            out["ok"] = true;
            out["user"]["username"] = rec.username;
            out["user"]["role"] = rec.role;
            out["user"]["email"] = rec.email;
            out["user"]["emailVerified"] = rec.emailVerified;
            if (enableEmailVerifyFlow) {
                std::string verifyToken;
                userStore.issueEmailVerifyToken(rec.username, verifyToken, 24LL * 60 * 60 * 1000);
                std::string returned;
                try {
                    returned = deliverToken(rec.email, "verify", verifyToken);
                } catch (...) {
                    returned.clear();
                }
                out["verifyRequired"] = true;
                if (!returned.empty()) out["verifyToken"] = returned;
            } else {
                rec.emailVerified = true;
                userStore.updateUser(rec);
                out["token"] = issueToken(rec);
            }
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            sendAuthErrorJson(cb, drogon::k500InternalServerError, "bootstrap exception", e.what());
            return;
        }
        cb(resp); });

    drogon::app().registerHandler("/auth/register", [&userStore, allowRegister, issueToken, enableEmailVerifyFlow, deliverToken](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            if (!allowRegister) {
                sendAuthErrorJson(cb, drogon::k403Forbidden, "register disabled");
                return;
            }
            auto json = req->getJsonObject();
            if (!json || !json->isMember("username") || !json->isMember("password") || !json->isMember("email")) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, "Missing username, password or email");
                return;
            }
            std::string err;
            const std::string username = (*json)["username"].asString();
            const std::string password = (*json)["password"].asString();
            const std::string email = (*json)["email"].asString();
            if (!userStore.addUser(username, email, password, "user", err)) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, err.empty() ? std::string("register failed") : err);
                return;
            }
            UserRecord rec;
            userStore.getUser(username, rec);
            Json::Value out;
            out["ok"] = true;
            out["user"]["username"] = rec.username;
            out["user"]["role"] = rec.role;
            out["user"]["email"] = rec.email;
            out["user"]["emailVerified"] = rec.emailVerified;
            if (enableEmailVerifyFlow) {
                std::string verifyToken;
                userStore.issueEmailVerifyToken(rec.username, verifyToken, 24LL * 60 * 60 * 1000);
                std::string returned;
                try {
                    returned = deliverToken(rec.email, "verify", verifyToken);
                } catch (...) {
                    returned.clear();
                }
                out["verifyRequired"] = true;
                if (!returned.empty()) out["verifyToken"] = returned;
            } else {
                rec.emailVerified = true;
                userStore.updateUser(rec);
                out["token"] = issueToken(rec);
            }
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            sendAuthErrorJson(cb, drogon::k500InternalServerError, "register exception", e.what());
            return;
        } catch (...) {
            sendAuthErrorJson(cb, drogon::k500InternalServerError, "register exception", "unknown exception");
            return;
        }
        cb(resp); });

    drogon::app().registerHandler("/auth/login", [&userStore, issueToken, enableEmailVerifyFlow](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("username") || !json->isMember("password")) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, "Missing username or password");
                return;
            }

            const std::string username = (*json)["username"].asString();
            const std::string password = (*json)["password"].asString();

            UserRecord user;
            if (!userStore.verifyUser(username, password, user)) {
                sendAuthErrorJson(cb, drogon::k401Unauthorized, "Invalid credentials");
                return;
            }

            if (enableEmailVerifyFlow && !user.emailVerified) {
                sendAuthErrorJson(cb, drogon::k403Forbidden, "email not verified");
                return;
            }

            auto token = issueToken(user);
            Json::Value out;
            out["ok"] = true;
            out["token"] = token;
            out["user"]["username"] = user.username;
            out["user"]["role"] = user.role;
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            sendAuthErrorJson(cb, drogon::k500InternalServerError, "login exception", e.what());
            return;
        }
        cb(resp); });

    drogon::app().registerHandler("/auth/me", [resolveAuthenticatedUser](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        UserRecord full;
        if (!resolveAuthenticatedUser(req, full)) {
            sendAuthErrorJson(cb, drogon::k401Unauthorized, "unauthorized");
            return;
        }
        Json::Value out;
        out["ok"] = true;
        out["user"] = userRecordToJson(full);
        resp->setStatusCode(drogon::k200OK);
        resp->setContentTypeString("application/json");
        resp->setBody(out.toStyledString());
        cb(resp); });

    drogon::app().registerHandler("/auth/profile", [resolveAuthenticatedUser, &userStore, enableEmailVerifyFlow, deliverToken](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        UserRecord full;
        if (!resolveAuthenticatedUser(req, full)) {
            sendAuthErrorJson(cb, drogon::k401Unauthorized, "unauthorized");
            return;
        }
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("email")) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, "Missing email");
                return;
            }
            std::string err;
            const std::string email = (*json)["email"].asString();
            if (!userStore.updateEmail(full.username, email, !enableEmailVerifyFlow, err)) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, err.empty() ? std::string("profile update failed") : err);
                return;
            }
            if (!userStore.getUser(full.username, full)) {
                sendAuthErrorJson(cb, drogon::k500InternalServerError, "profile refresh failed");
                return;
            }
            Json::Value out;
            out["ok"] = true;
            out["user"] = userRecordToJson(full);
            if (enableEmailVerifyFlow && !full.emailVerified) {
                std::string verifyToken;
                userStore.issueEmailVerifyToken(full.username, verifyToken, 24LL * 60 * 60 * 1000);
                std::string returned = deliverToken(full.email, "verify", verifyToken);
                out["verifyRequired"] = true;
                if (!returned.empty())
                    out["verifyToken"] = returned;
            }
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            sendAuthErrorJson(cb, drogon::k500InternalServerError, "profile exception", e.what());
            return;
        }
        cb(resp); }, {drogon::Patch});

    drogon::app().registerHandler("/auth/change-password", [resolveAuthenticatedUser, &userStore, issueToken](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        UserRecord full;
        if (!resolveAuthenticatedUser(req, full)) {
            sendAuthErrorJson(cb, drogon::k401Unauthorized, "unauthorized");
            return;
        }
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("oldPassword") || !json->isMember("newPassword")) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, "Missing oldPassword or newPassword");
                return;
            }
            std::string err;
            if (!userStore.changePassword(full.username, (*json)["oldPassword"].asString(), (*json)["newPassword"].asString(), err)) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, err.empty() ? std::string("change password failed") : err);
                return;
            }
            if (!userStore.getUser(full.username, full)) {
                sendAuthErrorJson(cb, drogon::k500InternalServerError, "password refresh failed");
                return;
            }
            Json::Value out;
            out["ok"] = true;
            out["token"] = issueToken(full);
            out["user"] = userRecordToJson(full);
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            sendAuthErrorJson(cb, drogon::k500InternalServerError, "change password exception", e.what());
            return;
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/auth/verify/request", [&userStore, deliverToken](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || (!json->isMember("username") && !json->isMember("email"))) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, "Missing username or email");
                return;
            }
            UserRecord rec;
            bool ok = false;
            if (json->isMember("username")) {
                ok = userStore.getUser((*json)["username"].asString(), rec);
            } else {
                ok = userStore.getUserByEmail((*json)["email"].asString(), rec);
            }
            if (!ok) {
                sendAuthErrorJson(cb, drogon::k404NotFound, "user not found");
                return;
            }
            std::string verifyToken;
            userStore.issueEmailVerifyToken(rec.username, verifyToken, 24LL * 60 * 60 * 1000);
            std::string returned = deliverToken(rec.email, "verify", verifyToken);
            Json::Value out;
            out["ok"] = true;
            if (!returned.empty()) out["verifyToken"] = returned;
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            sendAuthErrorJson(cb, drogon::k500InternalServerError, "verify request exception", e.what());
            return;
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/auth/verify", [&userStore](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("token") || (!json->isMember("username") && !json->isMember("email"))) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, "Missing token or username/email");
                return;
            }
            UserRecord rec;
            bool ok = false;
            if (json->isMember("username")) {
                ok = userStore.getUser((*json)["username"].asString(), rec);
            } else {
                ok = userStore.getUserByEmail((*json)["email"].asString(), rec);
            }
            if (!ok) {
                sendAuthErrorJson(cb, drogon::k404NotFound, "user not found");
                return;
            }
            const std::string token = (*json)["token"].asString();
            if (!userStore.verifyEmailToken(rec.username, token)) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, "invalid token");
                return;
            }
            Json::Value out;
            out["ok"] = true;
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            sendAuthErrorJson(cb, drogon::k500InternalServerError, "verify exception", e.what());
            return;
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/auth/forgot", [&userStore, deliverToken](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("email")) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, "Missing email");
                return;
            }
            std::string token;
            if (!userStore.issueResetTokenByEmail((*json)["email"].asString(), token, 60LL * 60 * 1000)) {
                sendAuthErrorJson(cb, drogon::k404NotFound, "email not found");
                return;
            }
            std::string returned = deliverToken((*json)["email"].asString(), "reset", token);
            Json::Value out;
            out["ok"] = true;
            if (!returned.empty()) out["resetToken"] = returned;
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            sendAuthErrorJson(cb, drogon::k500InternalServerError, "forgot exception", e.what());
            return;
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/auth/reset", [&userStore](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("email") || !json->isMember("token") || !json->isMember("password")) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, "Missing email, token or password");
                return;
            }
            if (!userStore.resetPasswordByEmail((*json)["email"].asString(), (*json)["token"].asString(), (*json)["password"].asString())) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, "reset failed");
                return;
            }
            Json::Value out;
            out["ok"] = true;
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            sendAuthErrorJson(cb, drogon::k500InternalServerError, "reset exception", e.what());
            return;
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/auth/logout", [](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        std::string token = extractBearerToken(req);
        if (token.empty()) {
            sendAuthErrorJson(cb, drogon::k401Unauthorized, "unauthorized");
            return;
        }
        auto resp = drogon::HttpResponse::newHttpResponse();
        Json::Value out;
        out["ok"] = true;
        out["revokedLocalToken"] = revokeLocalAuthToken(token);
        out["statelessJwt"] = (token.rfind("local-", 0) != 0);
        resp->setStatusCode(drogon::k200OK);
        resp->setContentTypeString("application/json");
        resp->setBody(out.toStyledString());
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/auth/admin/users", [resolveAuthenticatedUser, &userStore](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        UserRecord full;
        if (!resolveAuthenticatedUser(req, full)) {
            sendAuthErrorJson(cb, drogon::k401Unauthorized, "unauthorized");
            return;
        }
        if (full.role != "admin") {
            sendAuthErrorJson(cb, drogon::k403Forbidden, "admin only");
            return;
        }
        int limit = 100;
        std::string limitRaw = req->getParameter("limit");
        if (!limitRaw.empty()) {
            try {
                limit = std::max(1, std::min(1000, std::stoi(limitRaw)));
            } catch (...) {
                limit = 100;
            }
        }
        const bool includeTestUsers = req->getParameter("includeTestUsers") == "true";
        Json::Value users = userStore.listUsersSummary(limit, includeTestUsers);
        Json::Value out;
        out["ok"] = true;
        out["users"] = users;
        out["count"] = (Json::UInt64)users.size();
        resp->setStatusCode(drogon::k200OK);
        resp->setContentTypeString("application/json");
        resp->setBody(out.toStyledString());
        cb(resp); }, {drogon::Get});

    drogon::app().registerHandler("/auth/admin/cleanup-test-users", [resolveAuthenticatedUser, &userStore](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        UserRecord full;
        if (!resolveAuthenticatedUser(req, full)) {
            sendAuthErrorJson(cb, drogon::k401Unauthorized, "unauthorized");
            return;
        }
        if (full.role != "admin") {
            sendAuthErrorJson(cb, drogon::k403Forbidden, "admin only");
            return;
        }
        try {
            auto json = req->getJsonObject();
            bool dryRun = json && (*json).get("dryRun", false).asBool();
            std::vector<std::string> prefixes = {"autotest_", "bench_"};
            if (json && json->isMember("prefixes") && (*json)["prefixes"].isArray()) {
                prefixes.clear();
                for (const auto &item : (*json)["prefixes"]) {
                    if (item.isString() && !item.asString().empty())
                        prefixes.push_back(item.asString());
                }
            }
            if (prefixes.empty()) {
                sendAuthErrorJson(cb, drogon::k400BadRequest, "no prefixes supplied");
                return;
            }
            Json::Value removed(Json::arrayValue);
            size_t removedCount = userStore.cleanupUsersByPrefixes(prefixes, dryRun, removed);
            Json::Value out;
            out["ok"] = true;
            out["dryRun"] = dryRun;
            out["removedCount"] = (Json::UInt64)removedCount;
            out["removedUsers"] = removed;
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            sendAuthErrorJson(cb, drogon::k500InternalServerError, "cleanup exception", e.what());
            return;
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/api/gguf/inspect", [](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        const std::string modelPath = req->getParameter("path");
        if (modelPath.empty()) {
            resp->setStatusCode(drogon::k400BadRequest);
            resp->setContentTypeString("application/json");
            resp->setBody(nlohmann::json{{"ok", false}, {"error", "missing path"}}.dump(2));
            cb(resp);
            return;
        }
        auto inspection = gguf_tensor_parser::inspectFile(modelPath);
        nlohmann::json out{{"ok", inspection.valid}, {"result", inspection.toJson()}};
        if (!inspection.valid)
            out["error"] = inspection.error.empty() ? "gguf inspect failed" : inspection.error;
        resp->setStatusCode(inspection.valid ? drogon::k200OK : (inspection.exists ? drogon::k422UnprocessableEntity : drogon::k404NotFound));
        resp->setContentTypeString("application/json");
        resp->setBody(out.dump(2));
        cb(resp); }, {drogon::Get});

    drogon::app().registerHandler("/api/gguf/export", [](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        const std::string modelPath = req->getParameter("path");
        if (modelPath.empty()) {
            resp->setStatusCode(drogon::k400BadRequest);
            resp->setContentTypeString("application/json");
            resp->setBody(nlohmann::json{{"ok", false}, {"error", "missing path"}}.dump(2));
            cb(resp);
            return;
        }

        auto inspection = gguf_tensor_parser::inspectFile(modelPath);
        if (!inspection.valid) {
            nlohmann::json out{{"ok", false}, {"result", inspection.toJson()}, {"error", inspection.error.empty() ? "gguf export failed" : inspection.error}};
            resp->setStatusCode(inspection.exists ? drogon::k422UnprocessableEntity : drogon::k404NotFound);
            resp->setContentTypeString("application/json");
            resp->setBody(out.dump(2));
            cb(resp);
            return;
        }

        const auto bundle = gguf_tensor_parser::buildStructuredExportBundle("manual", modelPath, inspection, trantor::Date::now().microSecondsSinceEpoch() / 1000);
        const std::string outputDir = req->getParameter("outputDir");
        nlohmann::json out{{"ok", true}, {"result", bundle}};
        if (!outputDir.empty()) {
            std::string writeError;
            const auto manifest = gguf_tensor_parser::writeStructuredExportFiles(bundle, outputDir, &writeError);
            out["manifest"] = manifest;
            if (!writeError.empty()) {
                out["ok"] = false;
                out["error"] = writeError;
                resp->setStatusCode(drogon::k500InternalServerError);
            }
        }
        if (resp->statusCode() == drogon::kUnknown) {
            resp->setStatusCode(drogon::k200OK);
        }
        resp->setContentTypeString("application/json");
        resp->setBody(out.dump(2));
        cb(resp); }, {drogon::Get});

    auto platformPayload = [](const drogon::HttpRequestPtr &req) {
        auto body = req->getJsonObject();
        return body ? jsonCppToNlohmann(*body) : nlohmann::json::object();
    };

    drogon::app().registerHandler("/api/platform/status", [](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        sendNlohmannJson(cb, platformManager.status()); }, {drogon::Get});

    drogon::app().registerHandler("/api/platform/refresh", [](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        sendNlohmannJson(cb, platformManager.refreshTopology()); }, {drogon::Post});

    drogon::app().registerHandler("/api/platform/budget", [](const drogon::HttpRequestPtr &, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        try {
            const fs::path budgetPath = fs::current_path() / "build" / "stimulation_gerber" / "catastrophe1_npu_throughput_budget.json";
            if (!fs::exists(budgetPath) || !fs::is_regular_file(budgetPath)) {
                sendNlohmannJson(cb, nlohmann::json{{"ok", false}, {"error", "platform budget missing"}, {"path", budgetPath.string()}}, drogon::k404NotFound);
                return;
            }

            const auto report = loadNlohmannJsonFile(budgetPath);
            sendNlohmannJson(cb, nlohmann::json{{"ok", true}, {"path", budgetPath.string()}, {"report", report}});
        } catch (const std::exception &e) {
            sendNlohmannJson(cb, nlohmann::json{{"ok", false}, {"error", "platform budget exception"}, {"message", e.what()}}, drogon::k500InternalServerError);
        }
    }, {drogon::Get});

    drogon::app().registerHandler("/api/platform/config", [platformPayload](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        try {
            const auto payload = platformPayload(req);
            sendNlohmannJson(cb, platformManager.applyPatch(payload));
        } catch (const std::exception &e) {
            sendNlohmannJson(cb, nlohmann::json{{"ok", false}, {"error", "platform config exception"}, {"message", e.what()}}, drogon::k500InternalServerError);
        }
    }, {drogon::Patch});

    drogon::app().registerHandler("/api/platform/plan", [platformPayload](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        try {
            sendNlohmannJson(cb, platformManager.planCompute(platformPayload(req)));
        } catch (const std::exception &e) {
            sendNlohmannJson(cb, nlohmann::json{{"ok", false}, {"error", "platform plan exception"}, {"message", e.what()}}, drogon::k500InternalServerError);
        }
    }, {drogon::Post});

    drogon::app().registerHandler("/api/platform/dispatch/compute", [platformPayload](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        try {
            const auto out = platformManager.dispatchCompute(platformPayload(req));
            sendNlohmannJson(cb, out, out.value("accepted", false) ? drogon::k200OK : drogon::k409Conflict);
        } catch (const std::exception &e) {
            sendNlohmannJson(cb, nlohmann::json{{"ok", false}, {"error", "platform compute dispatch exception"}, {"message", e.what()}}, drogon::k500InternalServerError);
        }
    }, {drogon::Post});

    drogon::app().registerHandlerViaRegex("^/api/.*$", [proxyApiCall](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                          {
        proxyApiCall(req, std::move(cb)); });

    drogon::app().registerHandlerViaRegex("^/robots/.*$", [proxyApiCall](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                    {
        proxyApiCall(req, std::move(cb)); });

    drogon::app().registerHandler("/context/ingest", [&contextService, &v51Runtime, &worldModel, shouldRunV51, buildV51Request](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("text")) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("Missing text");
                cb(resp);
                return;
            }
            std::string sessionId;
            if (json->isMember("sessionId")) {
                sessionId = (*json)["sessionId"].asString();
            }
            if (sessionId.empty()) {
                sessionId = ContextService::generateSessionId();
            }
            std::string mode = "auto";
            if (json->isMember("mode")) {
                mode = (*json)["mode"].asString();
            }
            const std::string text = (*json)["text"].asString();
            auto result = contextService.ingest(sessionId, text, mode);
            Json::Value out;
            out["ok"] = true;
            out["context"] = result;
            out["sessionId"] = sessionId;
            auto worldResult = worldModel.ingestEvidence(nlohmann::json{{"sessionId", sessionId},
                                                                        {"modality", "text"},
                                                                        {"text", text},
                                                                        {"graphSummary", result.isMember("context") ? result["context"].asString() : std::string()},
                                                                        {"metadata", nlohmann::json{{"source", "context/ingest"},
                                                                                                    {"mode", mode},
                                                                                                    {"tokenCount", result.isMember("tokenCount") ? result["tokenCount"].asInt() : 0}}}});
            out["worldModel"] = nlohmannToJsonCpp(worldResult);
            Json::Value worldContexts(Json::arrayValue);
            auto ingestOptionalContext = [&](const char *fieldName, const char *modality)
            {
                if (!json->isMember(fieldName) || !(*json)[fieldName].isString())
                    return;
                const std::string contextValue = (*json)[fieldName].asString();
                if (contextValue.empty())
                    return;
                auto contextIngest = worldModel.ingestEvidence(nlohmann::json{{"sessionId", sessionId},
                                                                               {"modality", modality},
                                                                               {"graphSummary", contextValue},
                                                                               {"text", contextValue},
                                                                               {"metadata", nlohmann::json{{"source", "context/ingest"},
                                                                                                           {"field", fieldName},
                                                                                                           {"mode", mode},
                                                                                                           {"encoding", std::string("v-jpea2")}}}});
                worldContexts.append(nlohmannToJsonCpp(contextIngest));
            };
            ingestOptionalContext("videoContext", "video");
            ingestOptionalContext("imageContext", "vision");
            ingestOptionalContext("speechContext", "speech");

            if (json->isMember("videoContextVjpea2") && (*json)["videoContextVjpea2"].isObject())
            {
                auto vjpea2 = jsonCppToNlohmann((*json)["videoContextVjpea2"]);
                nlohmann::json metadata{{"source", "context/ingest"},
                                        {"field", "videoContextVjpea2"},
                                        {"mode", mode},
                                        {"encoding", std::string("v-jpea2")},
                                        {"vjpea2", vjpea2}};
                auto appendStringMetadata = [&](const char *fieldName)
                {
                    if (json->isMember(fieldName) && (*json)[fieldName].isString())
                        metadata[fieldName] = (*json)[fieldName].asString();
                };
                auto appendNumericMetadata = [&](const char *fieldName)
                {
                    if (json->isMember(fieldName) && (*json)[fieldName].isNumeric())
                        metadata[fieldName] = (*json)[fieldName].asDouble();
                };
                appendStringMetadata("cameraInterface");
                appendStringMetadata("sensorInterface");
                appendStringMetadata("captureBus");
                appendStringMetadata("cameraBus");
                appendStringMetadata("transport");
                appendNumericMetadata("frameRate");
                appendNumericMetadata("latencyMs");
                if (json->isMember("videoTimeline") && (*json)["videoTimeline"].isArray())
                {
                    metadata["videoTimeline"] = jsonCppToNlohmann((*json)["videoTimeline"]);
                }
                std::string summary;
                if (vjpea2.contains("focus") && vjpea2["focus"].is_string())
                    summary = vjpea2["focus"].get<std::string>();
                else if (vjpea2.contains("medium") && vjpea2["medium"].is_string())
                    summary = vjpea2["medium"].get<std::string>();
                else if (vjpea2.contains("coarse") && vjpea2["coarse"].is_string())
                    summary = vjpea2["coarse"].get<std::string>();

                auto contextIngest = worldModel.ingestEvidence(nlohmann::json{{"sessionId", sessionId},
                                                                               {"modality", "video"},
                                                                               {"graphSummary", summary},
                                                                               {"text", summary},
                                                                               {"metadata", metadata}});
                worldContexts.append(nlohmannToJsonCpp(contextIngest));
            }
            if (!worldContexts.empty())
                out["worldModelContexts"] = worldContexts;

            if (shouldRunV51(json.get())) {
                Json::Value reqV51 = buildV51Request(json.get(), sessionId, text);
                out["v51"] = v51Runtime.process(reqV51);
            } else {
                out["v51"] = Json::Value(Json::nullValue);
            }

            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(e.what());
        }
        cb(resp); });

    drogon::app().registerHandler("/v51/chat", [&v51Runtime, &contextService, &worldModel, shouldRunV51, buildV51Request](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("text")) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                Json::Value out;
                out["ok"] = false;
                out["error"] = "Missing text";
                resp->setBody(out.toStyledString());
                cb(resp);
                return;
            }
            std::string sessionId;
            if (json->isMember("sessionId"))
                sessionId = (*json)["sessionId"].asString();
            if (sessionId.empty())
                sessionId = ContextService::generateSessionId();

            std::string mode = "auto";
            if (json->isMember("mode"))
                mode = (*json)["mode"].asString();

            const std::string text = (*json)["text"].asString();
            auto context = contextService.ingest(sessionId, text, mode);

            Json::Value out;
            out["ok"] = true;
            out["sessionId"] = sessionId;
            out["context"] = context;
            auto worldResult = worldModel.ingestEvidence(nlohmann::json{{"sessionId", sessionId},
                                                                        {"modality", "text"},
                                                                        {"text", text},
                                                                        {"graphSummary", context.isMember("context") ? context["context"].asString() : std::string()},
                                                                        {"metadata", nlohmann::json{{"source", "v51/chat"},
                                                                                                    {"mode", mode},
                                                                                                    {"feedbackPresent", json->isMember("feedback")}}}});
            out["worldModel"] = nlohmannToJsonCpp(worldResult);

            if (shouldRunV51(json.get())) {
                Json::Value v51Req = buildV51Request(json.get(), sessionId, text);
                auto v51Result = v51Runtime.process(v51Req);
                out["v51"] = v51Result;
                if (json->isMember("feedback") && (*json)["feedback"].isNumeric()) {
                    Json::Value learnReq;
                    learnReq["sessionId"] = sessionId;
                    learnReq["feedback"] = (*json)["feedback"].asDouble();
                    if (json->isMember("learningRate") && (*json)["learningRate"].isNumeric())
                        learnReq["learningRate"] = (*json)["learningRate"].asDouble();
                    if (json->isMember("keywords") && (*json)["keywords"].isArray())
                        learnReq["keywords"] = (*json)["keywords"];
                    out["learn"] = v51Runtime.learn(learnReq);
                }
            } else {
                out["v51"] = Json::Value(Json::nullValue);
            }

            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            Json::Value out;
            out["ok"] = false;
            out["error"] = "v51 chat exception";
            out["message"] = e.what();
            resp->setBody(out.toStyledString());
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/context/reset", [&contextService, &worldModel](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("sessionId")) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("Missing sessionId");
                cb(resp);
                return;
            }
            const std::string sessionId = (*json)["sessionId"].asString();
            bool removed = contextService.reset(sessionId);
            bool worldRemoved = worldModel.resetSession(sessionId);
            Json::Value out;
            out["sessionId"] = sessionId;
            out["removed"] = removed;
            out["worldRemoved"] = worldRemoved;
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(e.what());
        }
        cb(resp); });

    drogon::app().registerHandler("/context/status", [&contextService](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto sessionId = req->getParameter("sessionId");
            if (sessionId.empty()) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("Missing sessionId");
                cb(resp);
                return;
            }
            auto result = contextService.status(sessionId);
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(result.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(e.what());
        }
        cb(resp); });

    drogon::app().registerHandler("/context/hint", [&contextService](const drogon::HttpRequestPtr &req,
                                                                    std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("text")) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("Missing text");
                cb(resp);
                return;
            }
            std::string sessionId;
            if (json->isMember("sessionId") && (*json)["sessionId"].isString())
                sessionId = (*json)["sessionId"].asString();
            std::string mode = "auto";
            if (json->isMember("mode") && (*json)["mode"].isString())
                mode = (*json)["mode"].asString();
            const std::string text = (*json)["text"].asString();
            std::string hint = contextService.prepareChatContext(sessionId, text, mode);
            Json::Value out;
            out["ok"] = true;
            out["contextHint"] = hint;
            out["mode"] = mode;
            out["weight"] = 0.9;
            out["sessionId"] = sessionId;
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(e.what());
        }
        cb(resp);
    }, {drogon::Post});

    drogon::app().registerHandler("/context/record", [&contextService](const drogon::HttpRequestPtr &req,
                                                                      std::function<void(const drogon::HttpResponsePtr &)> &&cb) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("reply")) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("Missing reply");
                cb(resp);
                return;
            }
            std::string sessionId;
            if (json->isMember("sessionId") && (*json)["sessionId"].isString())
                sessionId = (*json)["sessionId"].asString();
            std::string reply;
            if ((*json)["reply"].isString())
                reply = (*json)["reply"].asString();
            if (!sessionId.empty() && !reply.empty())
                contextService.recordAssistantReply(sessionId, reply);
            Json::Value out;
            out["ok"] = true;
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(e.what());
        }
        cb(resp);
    }, {drogon::Post});

    drogon::app().registerHandler("/context/knowledge", [&contextService](const drogon::HttpRequestPtr & /*req*/, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto result = contextService.knowledgeStatus();
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(result.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(e.what());
        }
        cb(resp); }, {drogon::Get});

    drogon::app().registerHandler("/world/status", [&worldModel](const drogon::HttpRequestPtr & /*req*/, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(worldModel.status().dump(2));
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            nlohmann::json out{{"ok", false}, {"error", "world status exception"}, {"message", e.what()}};
            resp->setBody(out.dump(2));
        }
        cb(resp); }, {drogon::Get});

    drogon::app().registerHandler("/world/physics/status", [bullet3Root,
                                                              defaultWorldPhysicsEnabled,
                                                              defaultWorldPhysicsBackend,
                                                              defaultWorldPhysicsSubsteps,
                                                              defaultEarthMapRequest](const drogon::HttpRequestPtr & /*req*/, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            nlohmann::json out{{"ok", true},
                               {"physicsRuntime", physics_world::inspectBullet3Runtime(bullet3Root)},
                               {"defaults", {{"physicsEnabled", defaultWorldPhysicsEnabled},
                                              {"physicsBackend", defaultWorldPhysicsBackend},
                                              {"physicsSubsteps", defaultWorldPhysicsSubsteps},
                                              {"earthMap", defaultEarthMapRequest}}}};
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.dump(2));
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            nlohmann::json out{{"ok", false}, {"error", "world physics status exception"}, {"message", e.what()}};
            resp->setBody(out.dump(2));
        }
        cb(resp); }, {drogon::Get});

    drogon::app().registerHandler("/world/state", [&worldModel](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            const std::string sessionId = req->getParameter("sessionId");
            if (sessionId.empty()) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                nlohmann::json out{{"ok", false}, {"error", "Missing sessionId"}};
                resp->setBody(out.dump(2));
                cb(resp);
                return;
            }
            int limit = 8;
            try {
                const std::string rawLimit = req->getParameter("limit");
                if (!rawLimit.empty())
                    limit = std::stoi(rawLimit);
            } catch (...) {
                limit = 8;
            }
            if (limit < 1)
                limit = 1;
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(worldModel.sessionState(sessionId, static_cast<std::size_t>(limit)).dump(2));
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            nlohmann::json out{{"ok", false}, {"error", "world state exception"}, {"message", e.what()}};
            resp->setBody(out.dump(2));
        }
        cb(resp); }, {drogon::Get});

    drogon::app().registerHandler("/world/cognitive", [&worldModel](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            const std::string sessionId = req->getParameter("sessionId");
            if (sessionId.empty()) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                nlohmann::json out{{"ok", false}, {"error", "Missing sessionId"}};
                resp->setBody(out.dump(2));
                cb(resp);
                return;
            }
            int limit = 8;
            try {
                const std::string rawLimit = req->getParameter("limit");
                if (!rawLimit.empty())
                    limit = std::stoi(rawLimit);
            } catch (...) {
                limit = 8;
            }
            if (limit < 1)
                limit = 1;
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            auto state = worldModel.sessionState(sessionId, static_cast<std::size_t>(limit));
            const std::string profile = req->getParameter("profile");
            if (world_model::lowerCopy(world_model::trimCopy(profile)) == "structural") {
                world_model::BrainProfileOptions options;
                options.kind = world_model::BrainProfileKind::Structural;
                resp->setBody(world_model::buildBrainProfile(state, options).dump(2));
            } else if (world_model::lowerCopy(world_model::trimCopy(profile)) == "dual") {
                resp->setBody(world_model::buildBrainProfiles(state).dump(2));
            } else {
                resp->setBody(world_model::buildCognitiveState(state).dump(2));
            }
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            nlohmann::json out{{"ok", false}, {"error", "world cognitive exception"}, {"message", e.what()}};
            resp->setBody(out.dump(2));
        }
        cb(resp); }, {drogon::Get});

    drogon::app().registerHandler("/world/brain", [&worldModel](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            const std::string sessionId = req->getParameter("sessionId");
            if (sessionId.empty()) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                nlohmann::json out{{"ok", false}, {"error", "Missing sessionId"}};
                resp->setBody(out.dump(2));
                cb(resp);
                return;
            }
            int limit = 8;
            try {
                const std::string rawLimit = req->getParameter("limit");
                if (!rawLimit.empty())
                    limit = std::stoi(rawLimit);
            } catch (...) {
                limit = 8;
            }
            if (limit < 1)
                limit = 1;
            const std::string profile = world_model::lowerCopy(world_model::trimCopy(req->getParameter("profile")));
            auto state = worldModel.sessionState(sessionId, static_cast<std::size_t>(limit));

            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            if (profile == "functional" || profile == "application") {
                world_model::BrainProfileOptions options;
                options.kind = world_model::BrainProfileKind::Functional;
                resp->setBody(world_model::buildBrainProfile(state, options).dump(2));
            } else if (profile == "structural" || profile == "research") {
                world_model::BrainProfileOptions options;
                options.kind = world_model::BrainProfileKind::Structural;
                resp->setBody(world_model::buildBrainProfile(state, options).dump(2));
            } else {
                resp->setBody(world_model::buildBrainProfiles(state).dump(2));
            }
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            nlohmann::json out{{"ok", false}, {"error", "world brain exception"}, {"message", e.what()}};
            resp->setBody(out.dump(2));
        }
        cb(resp); }, {drogon::Get});

    drogon::app().registerHandler("/world/conscious-compute", [&worldModel](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            const std::string sessionId = req->getParameter("sessionId");
            if (sessionId.empty()) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                nlohmann::json out{{"ok", false}, {"error", "Missing sessionId"}};
                resp->setBody(out.dump(2));
                cb(resp);
                return;
            }
            int limit = 8;
            try {
                const std::string rawLimit = req->getParameter("limit");
                if (!rawLimit.empty())
                    limit = std::stoi(rawLimit);
            } catch (...) {
                limit = 8;
            }
            if (limit < 1)
                limit = 1;

            world_model::ConsciousComputeOptions options;
            try {
                const std::string rawStages = req->getParameter("maxStages");
                if (!rawStages.empty())
                    options.maxStages = static_cast<std::size_t>(std::max(1, std::stoi(rawStages)));
            } catch (...) {
            }
            try {
                const std::string rawPrompts = req->getParameter("maxHumanPrompts");
                if (!rawPrompts.empty())
                    options.maxHumanPrompts = static_cast<std::size_t>(std::max(1, std::stoi(rawPrompts)));
            } catch (...) {
            }
            try {
                const std::string rawMachine = req->getParameter("maxMachineSteps");
                if (!rawMachine.empty())
                    options.maxMachineSteps = static_cast<std::size_t>(std::max(1, std::stoi(rawMachine)));
            } catch (...) {
            }
            auto state = worldModel.sessionState(sessionId, static_cast<std::size_t>(limit));
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(world_model::buildConsciousComputePlan(state, options).dump(2));
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            nlohmann::json out{{"ok", false}, {"error", "world conscious compute exception"}, {"message", e.what()}};
            resp->setBody(out.dump(2));
        }
        cb(resp); }, {drogon::Get});

    drogon::app().registerHandler("/world/collective-compute", [&worldModel](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                nlohmann::json out{{"ok", false}, {"error", "Missing JSON body"}};
                resp->setBody(out.dump(2));
                cb(resp);
                return;
            }

            auto payload = jsonCppToNlohmann(*json);
            const std::string sessionId = payload.value("sessionId", std::string());
            if (sessionId.empty()) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                nlohmann::json out{{"ok", false}, {"error", "Missing sessionId"}};
                resp->setBody(out.dump(2));
                cb(resp);
                return;
            }

            std::size_t limit = 8;
            if (payload.contains("limit") && payload["limit"].is_number_unsigned())
                limit = payload["limit"].get<std::size_t>();
            else if (payload.contains("limit") && payload["limit"].is_number_integer())
                limit = static_cast<std::size_t>(std::max<int64_t>(1, payload["limit"].get<int64_t>()));

            world_model::CollectiveComputeOptions options;
            if (payload.contains("participantCount") && payload["participantCount"].is_number_unsigned())
                options.participantCount = payload["participantCount"].get<std::size_t>();
            else if (payload.contains("participantCount") && payload["participantCount"].is_number_integer())
                options.participantCount = static_cast<std::size_t>(std::max<int64_t>(1, payload["participantCount"].get<int64_t>()));
            if (payload.contains("shardCount") && payload["shardCount"].is_number_unsigned())
                options.shardCount = payload["shardCount"].get<std::size_t>();
            else if (payload.contains("shardCount") && payload["shardCount"].is_number_integer())
                options.shardCount = static_cast<std::size_t>(std::max<int64_t>(1, payload["shardCount"].get<int64_t>()));
            if (payload.contains("redundancyFactor") && payload["redundancyFactor"].is_number_unsigned())
                options.redundancyFactor = payload["redundancyFactor"].get<std::size_t>();
            else if (payload.contains("redundancyFactor") && payload["redundancyFactor"].is_number_integer())
                options.redundancyFactor = static_cast<std::size_t>(std::max<int64_t>(1, payload["redundancyFactor"].get<int64_t>()));
            if (payload.contains("maxMnemonicWords") && payload["maxMnemonicWords"].is_number_unsigned())
                options.maxMnemonicWords = payload["maxMnemonicWords"].get<std::size_t>();
            else if (payload.contains("maxMnemonicWords") && payload["maxMnemonicWords"].is_number_integer())
                options.maxMnemonicWords = static_cast<std::size_t>(std::max<int64_t>(1, payload["maxMnemonicWords"].get<int64_t>()));
            if (payload.contains("maxCandidatesPerShard") && payload["maxCandidatesPerShard"].is_number_unsigned())
                options.maxCandidatesPerShard = payload["maxCandidatesPerShard"].get<std::size_t>();
            else if (payload.contains("maxCandidatesPerShard") && payload["maxCandidatesPerShard"].is_number_integer())
                options.maxCandidatesPerShard = static_cast<std::size_t>(std::max<int64_t>(1, payload["maxCandidatesPerShard"].get<int64_t>()));
            if (payload.contains("acceptableRelativeError") && payload["acceptableRelativeError"].is_number())
                options.acceptableRelativeError = std::max(0.0, std::min(1.0, payload["acceptableRelativeError"].get<double>()));
            if (payload.contains("allowMnemonicEncoding") && payload["allowMnemonicEncoding"].is_boolean())
                options.allowMnemonicEncoding = payload["allowMnemonicEncoding"].get<bool>();
            if (payload.contains("requireConsensus") && payload["requireConsensus"].is_boolean())
                options.requireConsensus = payload["requireConsensus"].get<bool>();

            auto state = worldModel.sessionState(sessionId, std::max<std::size_t>(1, limit));
            const auto computeTask = (payload.contains("computeTask") && payload["computeTask"].is_object()) ? payload["computeTask"] : nlohmann::json::object();
            const auto plan = world_model::buildCollectiveConsciousComputePlan(state, computeTask, options);

            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(plan.dump(2));
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            nlohmann::json out{{"ok", false}, {"error", "world collective compute exception"}, {"message", e.what()}};
            resp->setBody(out.dump(2));
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/world/ingest", [&worldModel](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                nlohmann::json out{{"ok", false}, {"error", "Missing JSON body"}};
                resp->setBody(out.dump(2));
                cb(resp);
                return;
            }
            auto payload = jsonCppToNlohmann(*json);
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(worldModel.ingestEvidence(std::move(payload)).dump(2));
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            nlohmann::json out{{"ok", false}, {"error", "world ingest exception"}, {"message", e.what()}};
            resp->setBody(out.dump(2));
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/world/earth-map/import", [&worldModel, bullet3Root](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                nlohmann::json out{{"ok", false}, {"error", "Missing JSON body"}};
                resp->setBody(out.dump(2));
                cb(resp);
                return;
            }
            auto payload = jsonCppToNlohmann(*json);
            const std::string sessionId = payload.value("sessionId", std::string());
            if (sessionId.empty()) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                nlohmann::json out{{"ok", false}, {"error", "Missing sessionId"}};
                resp->setBody(out.dump(2));
                cb(resp);
                return;
            }

            nlohmann::json mapRequest = payload.contains("earthMap") ? payload["earthMap"] : payload;
            auto manifest = physics_world::normalizeEarthMapImportRequest(mapRequest);
            const auto runtime = physics_world::inspectBullet3Runtime(bullet3Root);

            if (payload.value("persist", true)) {
                nlohmann::json ingestPayload{{"sessionId", sessionId},
                                             {"modality", "earth_map"},
                                             {"graphSummary", manifest.value("summary", std::string())},
                                             {"metadata", {{"source", "world/earth-map/import"},
                                                            {"sourceUri", manifest.value("sourceUri", std::string())},
                                                            {"format", manifest.value("format", std::string())},
                                                            {"coordinateFrame", manifest.value("coordinateFrame", std::string())},
                                                            {"regionLabel", manifest.value("regionLabel", std::string())},
                                                            {"lod", manifest.value("lod", 0)}}}};
                worldModel.ingestEvidence(std::move(ingestPayload));
            }

            nlohmann::json out{{"ok", true}, {"earthMap", manifest}, {"physicsRuntime", runtime}};
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.dump(2));
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            nlohmann::json out{{"ok", false}, {"error", "world earth-map import exception"}, {"message", e.what()}};
            resp->setBody(out.dump(2));
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/world/simulate", [&worldModel,
                                                       defaultWorldAgentCount,
                                                       defaultWorldMapWidth,
                                                       defaultWorldMapHeight,
                                                       defaultWorldMapDepth,
                                                       defaultWorldDialogueTurns,
                                                       defaultWorldEcologyClusters,
                                                       defaultWorld3DMap,
                                                       defaultWorldEmbodiedAgents,
                                                       defaultWorldEcologyVideo,
                                                       defaultWorldPhysicsEnabled,
                                                       defaultWorldPhysicsBackend,
                                                       defaultWorldPhysicsSubsteps,
                                                       defaultEarthMapRequest,
                                                       bullet3Root](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                nlohmann::json out{{"ok", false}, {"error", "Missing JSON body"}};
                resp->setBody(out.dump(2));
                cb(resp);
                return;
            }
            auto payload = jsonCppToNlohmann(*json);
            const std::string sessionId = payload.value("sessionId", std::string());
            if (sessionId.empty()) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                nlohmann::json out{{"ok", false}, {"error", "Missing sessionId"}};
                resp->setBody(out.dump(2));
                cb(resp);
                return;
            }

            std::size_t limit = 8;
            if (payload.contains("limit") && payload["limit"].is_number_unsigned())
                limit = payload["limit"].get<std::size_t>();
            else if (payload.contains("limit") && payload["limit"].is_number_integer())
                limit = static_cast<std::size_t>(std::max<int64_t>(1, payload["limit"].get<int64_t>()));

            world_model::VirtualSceneOptions options;
            options.maxAgents = static_cast<std::size_t>(defaultWorldAgentCount);
            options.mapWidth = static_cast<std::size_t>(defaultWorldMapWidth);
            options.mapHeight = static_cast<std::size_t>(defaultWorldMapHeight);
            options.mapDepth = static_cast<std::size_t>(defaultWorldMapDepth);
            options.maxDialogueTurns = static_cast<std::size_t>(defaultWorldDialogueTurns);
            options.maxEcologyClusters = static_cast<std::size_t>(defaultWorldEcologyClusters);
            options.include3DMap = defaultWorld3DMap;
            options.includeEmbodiedAgents = defaultWorldEmbodiedAgents;
            options.includeEcologyFromVideo = defaultWorldEcologyVideo;
            options.physicsEnabled = defaultWorldPhysicsEnabled;
            options.physicsBackend = defaultWorldPhysicsBackend;
            options.physicsSubsteps = static_cast<std::size_t>(defaultWorldPhysicsSubsteps);
            options.earthMapEnabled = defaultEarthMapRequest.value("enabled", false);
            options.earthMapRequest = defaultEarthMapRequest;
            if (payload.contains("maxAgents") && payload["maxAgents"].is_number_unsigned())
                options.maxAgents = payload["maxAgents"].get<std::size_t>();
            else if (payload.contains("maxAgents") && payload["maxAgents"].is_number_integer())
                options.maxAgents = static_cast<std::size_t>(std::max<int64_t>(1, payload["maxAgents"].get<int64_t>()));
            if (payload.contains("maxSteps") && payload["maxSteps"].is_number_unsigned())
                options.maxSteps = payload["maxSteps"].get<std::size_t>();
            else if (payload.contains("maxSteps") && payload["maxSteps"].is_number_integer())
                options.maxSteps = static_cast<std::size_t>(std::max<int64_t>(1, payload["maxSteps"].get<int64_t>()));
            if (payload.contains("mapWidth") && payload["mapWidth"].is_number_unsigned())
                options.mapWidth = payload["mapWidth"].get<std::size_t>();
            else if (payload.contains("mapWidth") && payload["mapWidth"].is_number_integer())
                options.mapWidth = static_cast<std::size_t>(std::max<int64_t>(2, payload["mapWidth"].get<int64_t>()));
            if (payload.contains("mapHeight") && payload["mapHeight"].is_number_unsigned())
                options.mapHeight = payload["mapHeight"].get<std::size_t>();
            else if (payload.contains("mapHeight") && payload["mapHeight"].is_number_integer())
                options.mapHeight = static_cast<std::size_t>(std::max<int64_t>(2, payload["mapHeight"].get<int64_t>()));
            if (payload.contains("mapDepth") && payload["mapDepth"].is_number_unsigned())
                options.mapDepth = payload["mapDepth"].get<std::size_t>();
            else if (payload.contains("mapDepth") && payload["mapDepth"].is_number_integer())
                options.mapDepth = static_cast<std::size_t>(std::max<int64_t>(1, payload["mapDepth"].get<int64_t>()));
            if (payload.contains("maxTrainSamples") && payload["maxTrainSamples"].is_number_unsigned())
                options.maxTrainSamples = payload["maxTrainSamples"].get<std::size_t>();
            else if (payload.contains("maxTrainSamples") && payload["maxTrainSamples"].is_number_integer())
                options.maxTrainSamples = static_cast<std::size_t>(std::max<int64_t>(1, payload["maxTrainSamples"].get<int64_t>()));
            if (payload.contains("maxEventChars") && payload["maxEventChars"].is_number_unsigned())
                options.maxEventChars = payload["maxEventChars"].get<std::size_t>();
            else if (payload.contains("maxEventChars") && payload["maxEventChars"].is_number_integer())
                options.maxEventChars = static_cast<std::size_t>(std::max<int64_t>(24, payload["maxEventChars"].get<int64_t>()));
            if (payload.contains("maxDialogueTurns") && payload["maxDialogueTurns"].is_number_unsigned())
                options.maxDialogueTurns = payload["maxDialogueTurns"].get<std::size_t>();
            else if (payload.contains("maxDialogueTurns") && payload["maxDialogueTurns"].is_number_integer())
                options.maxDialogueTurns = static_cast<std::size_t>(std::max<int64_t>(0, payload["maxDialogueTurns"].get<int64_t>()));
            if (payload.contains("maxEcologyClusters") && payload["maxEcologyClusters"].is_number_unsigned())
                options.maxEcologyClusters = payload["maxEcologyClusters"].get<std::size_t>();
            else if (payload.contains("maxEcologyClusters") && payload["maxEcologyClusters"].is_number_integer())
                options.maxEcologyClusters = static_cast<std::size_t>(std::max<int64_t>(0, payload["maxEcologyClusters"].get<int64_t>()));
            if (payload.contains("includeCounterfactual") && payload["includeCounterfactual"].is_boolean())
                options.includeCounterfactual = payload["includeCounterfactual"].get<bool>();
            if (payload.contains("includeSceneRecallSample") && payload["includeSceneRecallSample"].is_boolean())
                options.includeSceneRecallSample = payload["includeSceneRecallSample"].get<bool>();
            if (payload.contains("include3DMap") && payload["include3DMap"].is_boolean())
                options.include3DMap = payload["include3DMap"].get<bool>();
            if (payload.contains("includeEmbodiedAgents") && payload["includeEmbodiedAgents"].is_boolean())
                options.includeEmbodiedAgents = payload["includeEmbodiedAgents"].get<bool>();
            if (payload.contains("includeEcologyFromVideo") && payload["includeEcologyFromVideo"].is_boolean())
                options.includeEcologyFromVideo = payload["includeEcologyFromVideo"].get<bool>();
            if (payload.contains("physicsEnabled") && payload["physicsEnabled"].is_boolean())
                options.physicsEnabled = payload["physicsEnabled"].get<bool>();
            if (payload.contains("physicsBackend") && payload["physicsBackend"].is_string())
                options.physicsBackend = payload["physicsBackend"].get<std::string>();
            if (payload.contains("physicsSubsteps") && payload["physicsSubsteps"].is_number_unsigned())
                options.physicsSubsteps = payload["physicsSubsteps"].get<std::size_t>();
            else if (payload.contains("physicsSubsteps") && payload["physicsSubsteps"].is_number_integer())
                options.physicsSubsteps = static_cast<std::size_t>(std::max<int64_t>(1, payload["physicsSubsteps"].get<int64_t>()));
            if (payload.contains("earthMap") && (payload["earthMap"].is_object() || payload["earthMap"].is_string())) {
                options.earthMapRequest = physics_world::normalizeEarthMapImportRequest(payload["earthMap"]);
                options.earthMapEnabled = options.earthMapRequest.value("enabled", false);
            } else {
                if (payload.contains("earthMapEnabled") && payload["earthMapEnabled"].is_boolean())
                    options.earthMapEnabled = payload["earthMapEnabled"].get<bool>();
                if ((payload.contains("earthMapUri") && payload["earthMapUri"].is_string()) ||
                    (payload.contains("earthMapFormat") && payload["earthMapFormat"].is_string())) {
                    auto request = options.earthMapRequest.is_object() ? options.earthMapRequest : nlohmann::json::object();
                    request["enabled"] = options.earthMapEnabled;
                    if (payload.contains("earthMapUri") && payload["earthMapUri"].is_string())
                        request["sourceUri"] = payload["earthMapUri"].get<std::string>();
                    if (payload.contains("earthMapFormat") && payload["earthMapFormat"].is_string())
                        request["format"] = payload["earthMapFormat"].get<std::string>();
                    options.earthMapRequest = physics_world::normalizeEarthMapImportRequest(request);
                    options.earthMapEnabled = options.earthMapRequest.value("enabled", false);
                }
            }
            if (payload.contains("brainProfile") && payload["brainProfile"].is_string())
                options.brainProfile = world_model::parseBrainProfileKind(payload["brainProfile"].get<std::string>());

            auto state = worldModel.sessionState(sessionId, std::max<std::size_t>(1, limit));
            auto simulation = world_model::simulateVirtualScene(state, options);
            const auto physicsRuntime = physics_world::inspectBullet3Runtime(bullet3Root);
            auto physicsExecution = physics_world::executeNativePhysicsScene(
                simulation.contains("physicsScene") && simulation["physicsScene"].is_object() ? simulation["physicsScene"] : nlohmann::json::object(),
                fs::current_path(),
                std::max<std::size_t>(8, options.maxSteps * std::max<std::size_t>(2, options.maxDialogueTurns + 2)),
                1.0 / 12.0);
            simulation["physicsRuntime"] = physicsRuntime;
            physicsExecution["runtime"] = physicsRuntime;
            simulation["physicsExecution"] = physicsExecution;
            if (simulation.contains("physicsScene") && simulation["physicsScene"].is_object()) {
                simulation["physicsScene"]["runtime"] = physicsRuntime;
                simulation["physicsScene"]["executionStatus"] = physicsExecution.value("status", std::string("unknown"));
                if (physicsExecution.contains("summary") && physicsExecution["summary"].is_string()) {
                    simulation["physicsScene"]["executionSummary"] = physicsExecution["summary"];
                }
                if (physicsExecution.contains("terrain") && physicsExecution["terrain"].is_object()) {
                    simulation["physicsScene"]["executedTerrain"] = physicsExecution["terrain"];
                }
            }
            if (simulation.contains("trainSamples") && simulation["trainSamples"].is_array() &&
                physicsExecution.contains("trainSamples") && physicsExecution["trainSamples"].is_array()) {
                for (const auto &sample : physicsExecution["trainSamples"]) {
                    simulation["trainSamples"].push_back(sample);
                }
            }

            if (payload.value("persist", false) && simulation.contains("timeline") && simulation["timeline"].is_array()) {
                for (const auto &step : simulation["timeline"]) {
                    if (!step.is_object() || !step.contains("events") || !step["events"].is_array())
                        continue;
                    const int stepIndex = step.value("step", 0);
                    for (const auto &event : step["events"]) {
                        if (!event.is_object() || !event.contains("event") || !event["event"].is_string())
                            continue;
                        nlohmann::json ingestPayload{
                            {"sessionId", sessionId},
                            {"modality", "simulation"},
                            {"graphSummary", event["event"].get<std::string>()},
                            {"metadata", {
                                {"source", "world/simulate"},
                                {"agentId", event.value("agentId", std::string())},
                                {"role", event.value("role", std::string())},
                                {"focus", event.value("focus", std::string())},
                                {"step", stepIndex}
                            }}
                        };
                        worldModel.ingestEvidence(std::move(ingestPayload));
                    }
                    if (step.contains("dialogues") && step["dialogues"].is_array()) {
                        for (const auto &dialogue : step["dialogues"]) {
                            if (!dialogue.is_object() || !dialogue.contains("utterance") || !dialogue["utterance"].is_string())
                                continue;
                            nlohmann::json ingestPayload{
                                {"sessionId", sessionId},
                                {"modality", "simulation_dialogue"},
                                {"graphSummary", dialogue["utterance"].get<std::string>()},
                                {"metadata", {
                                    {"source", "world/simulate"},
                                    {"speakerId", dialogue.value("speakerId", std::string())},
                                    {"listenerId", dialogue.value("listenerId", std::string())},
                                    {"turn", dialogue.value("turn", 0)},
                                    {"step", stepIndex}
                                }}
                            };
                            worldModel.ingestEvidence(std::move(ingestPayload));
                        }
                    }
                    if (step.contains("environmentalSignals") && step["environmentalSignals"].is_array()) {
                        for (const auto &signal : step["environmentalSignals"]) {
                            if (!signal.is_object() || !signal.contains("summary") || !signal["summary"].is_string())
                                continue;
                            nlohmann::json ingestPayload{
                                {"sessionId", sessionId},
                                {"modality", "simulation_ecology"},
                                {"graphSummary", signal["summary"].get<std::string>()},
                                {"metadata", {
                                    {"source", "world/simulate"},
                                    {"entityId", signal.value("entityId", std::string())},
                                    {"biomeLabel", signal.value("biomeLabel", std::string())},
                                    {"step", stepIndex}
                                }}
                            };
                            worldModel.ingestEvidence(std::move(ingestPayload));
                        }
                    }
                }
                if (simulation.contains("worldMap3D") && simulation["worldMap3D"].is_object() &&
                    simulation["worldMap3D"].contains("sceneEnvelope") && simulation["worldMap3D"]["sceneEnvelope"].is_object()) {
                    const auto &sceneEnvelope = simulation["worldMap3D"]["sceneEnvelope"];
                    if (sceneEnvelope.contains("summary") && sceneEnvelope["summary"].is_string()) {
                        nlohmann::json ingestPayload{
                            {"sessionId", sessionId},
                            {"modality", "simulation_map"},
                            {"graphSummary", sceneEnvelope["summary"].get<std::string>()},
                            {"metadata", {
                                {"source", "world/simulate"},
                                {"mapDepth", simulation["worldMap3D"].value("dimensions", nlohmann::json::object()).value("depth", 0)}
                            }}
                        };
                        worldModel.ingestEvidence(std::move(ingestPayload));
                    }
                }
                if (simulation.contains("physicsScene") && simulation["physicsScene"].is_object() &&
                    simulation["physicsScene"].contains("summary") && simulation["physicsScene"]["summary"].is_string()) {
                    nlohmann::json ingestPayload{{"sessionId", sessionId},
                                                 {"modality", "simulation_physics"},
                                                 {"graphSummary", simulation["physicsScene"]["summary"].get<std::string>()},
                                                 {"metadata", {{"source", "world/simulate"},
                                                                {"backend", simulation["physicsScene"].value("backend", std::string())},
                                                                {"substeps", simulation["physicsScene"].value("substeps", 0)}}}};
                    worldModel.ingestEvidence(std::move(ingestPayload));
                }
                if (simulation.contains("physicsExecution") && simulation["physicsExecution"].is_object() &&
                    simulation["physicsExecution"].contains("summary") && simulation["physicsExecution"]["summary"].is_string()) {
                    nlohmann::json ingestPayload{{"sessionId", sessionId},
                                                 {"modality", "simulation_physics_runtime"},
                                                 {"graphSummary", simulation["physicsExecution"]["summary"].get<std::string>()},
                                                 {"metadata", {{"source", "world/simulate"},
                                                                {"status", simulation["physicsExecution"].value("status", std::string())},
                                                                {"backend", simulation["physicsExecution"].value("backend", std::string())},
                                                                {"frames", simulation["physicsExecution"].value("frames", 0)}}}};
                    worldModel.ingestEvidence(std::move(ingestPayload));
                }
                if (simulation.contains("physicsExecution") && simulation["physicsExecution"].is_object() &&
                    simulation["physicsExecution"].contains("terrain") && simulation["physicsExecution"]["terrain"].is_object() &&
                    simulation["physicsExecution"]["terrain"].contains("summary") && simulation["physicsExecution"]["terrain"]["summary"].is_string()) {
                    const auto &terrain = simulation["physicsExecution"]["terrain"];
                    nlohmann::json ingestPayload{{"sessionId", sessionId},
                                                 {"modality", "simulation_terrain_runtime"},
                                                 {"graphSummary", terrain["summary"].get<std::string>()},
                                                 {"metadata", {{"source", "world/simulate"},
                                                                {"format", terrain.value("format", std::string())},
                                                                {"regionLabel", terrain.value("regionLabel", std::string())},
                                                                {"sourceStatus", terrain.value("sourceStatus", std::string())}}}};
                    worldModel.ingestEvidence(std::move(ingestPayload));
                }
                if (simulation.contains("physicsExecution") && simulation["physicsExecution"].is_object() &&
                    simulation["physicsExecution"].contains("bodySummaries") && simulation["physicsExecution"]["bodySummaries"].is_array()) {
                    std::size_t motionCount = 0;
                    for (const auto &body : simulation["physicsExecution"]["bodySummaries"]) {
                        if (!body.is_object() || !body.value("dynamic", false)) {
                            continue;
                        }
                        std::ostringstream motion;
                        motion << body.value("id", std::string("body"))
                               << " displaced " << std::fixed << std::setprecision(2) << body.value("displacementMeters", 0.0)
                               << "m with peak speed " << body.value("peakSpeedMps", 0.0) << "m/s";
                        nlohmann::json ingestPayload{{"sessionId", sessionId},
                                                     {"modality", "simulation_body_motion"},
                                                     {"graphSummary", motion.str()},
                                                     {"metadata", {{"source", "world/simulate"},
                                                                    {"bodyId", body.value("id", std::string())},
                                                                    {"role", body.value("role", std::string())}}}};
                        worldModel.ingestEvidence(std::move(ingestPayload));
                        ++motionCount;
                        if (motionCount >= 3) {
                            break;
                        }
                    }
                }
                if (simulation.contains("physicsScene") && simulation["physicsScene"].is_object() &&
                    simulation["physicsScene"].contains("earthMap") && simulation["physicsScene"]["earthMap"].is_object() &&
                    simulation["physicsScene"]["earthMap"].value("enabled", false) &&
                    simulation["physicsScene"]["earthMap"].contains("summary") && simulation["physicsScene"]["earthMap"]["summary"].is_string()) {
                    const auto &earthMap = simulation["physicsScene"]["earthMap"];
                    nlohmann::json ingestPayload{{"sessionId", sessionId},
                                                 {"modality", "simulation_earth_map"},
                                                 {"graphSummary", earthMap["summary"].get<std::string>()},
                                                 {"metadata", {{"source", "world/simulate"},
                                                                {"sourceUri", earthMap.value("sourceUri", std::string())},
                                                                {"format", earthMap.value("format", std::string())},
                                                                {"regionLabel", earthMap.value("regionLabel", std::string())}}}};
                    worldModel.ingestEvidence(std::move(ingestPayload));
                }
                simulation["persisted"] = true;
            }

            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(simulation.dump(2));
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            nlohmann::json out{{"ok", false}, {"error", "world simulate exception"}, {"message", e.what()}};
            resp->setBody(out.dump(2));
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/vision/analyze", [&imageWorldModel, &worldModel](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("Missing JSON body");
                cb(resp);
                return;
            }
            if (!imageWorldModel) {
                resp->setStatusCode(drogon::k500InternalServerError);
                resp->setBody("JPEA image world model is unavailable");
                cb(resp);
                return;
            }

            cv::Mat img;
            if (json->isMember("imageBase64")) {
                std::string b64 = (*json)["imageBase64"].asString();
                if (b64.rfind("data:", 0) == 0) {
                    auto pos = b64.find("base64,");
                    if (pos != std::string::npos) b64 = b64.substr(pos + 7);
                }
                thread_local std::vector<uint8_t> decodeBuf;
                base64DecodeTo(b64, decodeBuf);
                img = cv::imdecode(decodeBuf, cv::IMREAD_COLOR);
            } else if (json->isMember("imagePath")) {
                std::string path = (*json)["imagePath"].asString();
                img = cv::imread(path, cv::IMREAD_COLOR);
            }

            if (img.empty()) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("Invalid or empty image");
                cb(resp);
                return;
            }

            std::vector<uchar> encodedImage;
            if (!cv::imencode(".jpg", img, encodedImage)) {
                resp->setStatusCode(drogon::k500InternalServerError);
                resp->setBody("Failed to encode image for JPEA analysis");
                cb(resp);
                return;
            }
            const auto embedding = imageWorldModel->encode(
                std::vector<uint8_t>(encodedImage.begin(), encodedImage.end()),
                img.cols,
                img.rows,
                "image/jpeg");
            Json::Value result(Json::objectValue);
            result["ok"] = !embedding.empty();
            result["model"] = imageWorldModel->config().id;
            result["architecture"] = imageWorldModel->config().arch;
            result["backend"] = imageWorldModel->status().value("backend", std::string("unknown"));
            result["imageSize"]["w"] = img.cols;
            result["imageSize"]["h"] = img.rows;
            result["detections"] = Json::Value(Json::arrayValue);
            result["details"] = Json::Value(Json::arrayValue);
            result["embedding"] = Json::Value(Json::arrayValue);
            for (float value : embedding) {
                result["embedding"].append(value);
            }
            result["embeddingDim"] = static_cast<int>(embedding.size());
            result["graphContext"] = "vision-jepa|model:" + imageWorldModel->config().id + "|embeddingDim:" + std::to_string(embedding.size());
            if (embedding.empty()) {
                result["error"] = "JPEA image world model returned an empty concept vector";
            }
            std::string sessionId;
            if (json->isMember("sessionId")) {
                sessionId = (*json)["sessionId"].asString();
            }
            if (!sessionId.empty()) {
                nlohmann::json metadata{{"source", "vision/analyze"},
                                        {"detections", result.isMember("detections") ? jsonCppToNlohmann(result["detections"]) : nlohmann::json::array()},
                                        {"details", result.isMember("details") ? jsonCppToNlohmann(result["details"]) : nlohmann::json::array()}};
                if (result.isMember("embedding")) {
                    metadata["embedding"] = jsonCppToNlohmann(result["embedding"]);
                }
                if (result.isMember("graphContext") && result["graphContext"].isString()) {
                    metadata["graphContext"] = result["graphContext"].asString();
                }
                if (json->isMember("imagePath") && (*json)["imagePath"].isString()) {
                    metadata["imagePath"] = (*json)["imagePath"].asString();
                }
                auto worldResult = worldModel.ingestEvidence(nlohmann::json{{"sessionId", sessionId},
                                                                            {"modality", "vision"},
                                                                            {"graphSummary", result.isMember("graphContext") && result["graphContext"].isString() ? result["graphContext"].asString() : std::string()},
                                                                            {"rawLocation", json->isMember("imagePath") && (*json)["imagePath"].isString() ? (*json)["imagePath"].asString() : std::string("inline-image")},
                                                                            {"metadata", metadata}});
                result["worldModel"] = nlohmannToJsonCpp(worldResult);
            }
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(result.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(e.what());
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/camera/analyze", [&imageWorldModel, &worldModel](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("Missing JSON body");
                cb(resp);
                return;
            }
            if (!imageWorldModel || !imageWorldModel->status().value("ready", false)) {
                resp->setStatusCode(drogon::k503ServiceUnavailable);
                resp->setBody("compiled RDK X5 JPEA model is unavailable");
                cb(resp);
                return;
            }

            if (json->isMember("videoBase64") || json->isMember("videoPath")) {
                resp->setStatusCode(drogon::k410Gone);
                resp->setBody("host video transport is disabled; use the X5-local camera endpoint without videoBase64 or videoPath");
                cb(resp);
                return;
            }

            const std::string cameraDevice = resolveConfig<std::string>("jpea.camera.device", std::string("/dev/video0"), "JPEA_CAMERA_DEVICE");
            const int configuredWidth = std::max(1, resolveConfig<int>("jpea.camera.width", 1920, "JPEA_CAMERA_WIDTH"));
            const int configuredHeight = std::max(1, resolveConfig<int>("jpea.camera.height", 1080, "JPEA_CAMERA_HEIGHT"));
            const int configuredFps = std::max(1, resolveConfig<int>("jpea.camera.fps", 30, "JPEA_CAMERA_FPS"));
            cv::VideoCapture capture(cameraDevice, cv::CAP_V4L2);
            if (capture.isOpened()) {
                capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                capture.set(cv::CAP_PROP_FRAME_WIDTH, configuredWidth);
                capture.set(cv::CAP_PROP_FRAME_HEIGHT, configuredHeight);
                capture.set(cv::CAP_PROP_FPS, configuredFps);
            }
            cv::Mat frame;
            bool decoded = false;
            for (int attempt = 0; attempt < 12 && !decoded; ++attempt) {
                decoded = capture.isOpened() && capture.read(frame) && !frame.empty();
            }
            if (!decoded) {
                resp->setStatusCode(drogon::k503ServiceUnavailable);
                resp->setBody("X5 camera did not provide a frame from " + cameraDevice);
                cb(resp);
                return;
            }

            static std::mutex videoFormatMutex;
            static int lockedWidth = 0;
            static int lockedHeight = 0;
            static int lockedType = -1;
            {
                std::lock_guard<std::mutex> lock(videoFormatMutex);
                if (lockedWidth == 0) {
                    lockedWidth = configuredWidth;
                    lockedHeight = configuredHeight;
                    lockedType = frame.type();
                }
                if (frame.cols != lockedWidth || frame.rows != lockedHeight || frame.type() != lockedType) {
                    resp->setStatusCode(drogon::k409Conflict);
                    resp->setBody("video frame specification differs from the locked JPEA deployment specification");
                    cb(resp);
                    return;
                }
            }

            if (!frame.isContinuous() || frame.type() != CV_8UC3) {
                resp->setStatusCode(drogon::k500InternalServerError);
                resp->setBody("camera returned a non-contiguous or non-BGR frame");
                cb(resp);
                return;
            }
            const auto embedding = imageWorldModel->encode(std::vector<uint8_t>(frame.datastart, frame.dataend), frame.cols, frame.rows, "application/x-bgr");
            if (embedding.empty()) {
                resp->setStatusCode(drogon::k503ServiceUnavailable);
                resp->setBody(imageWorldModel->status().value("error", std::string("RDK X5 JPEA inference failed")));
                cb(resp);
                return;
            }
            Json::Value result(Json::objectValue);
            result["ok"] = true;
            result["backend"] = "horizon-hbdnn";
            result["model"] = imageWorldModel->config().id;
            result["frameSpec"]["width"] = frame.cols;
            result["frameSpec"]["height"] = frame.rows;
            result["frameSpec"]["opencvType"] = frame.type();
            result["embedding"] = Json::Value(Json::arrayValue);
            for (float value : embedding) result["embedding"].append(value);
            result["embeddingDim"] = static_cast<int>(embedding.size());
            const std::string sessionId = json->isMember("sessionId") ? (*json)["sessionId"].asString() : std::string();
            if (!sessionId.empty()) {
                nlohmann::json metadata{{"source", "video/analyze"}, {"frameWidth", frame.cols}, {"frameHeight", frame.rows}, {"embedding", embedding}, {"jpeaBackend", "horizon-hbdnn"}};
                result["worldModel"] = nlohmannToJsonCpp(worldModel.ingestEvidence(nlohmann::json{{"sessionId", sessionId}, {"modality", "video"}, {"graphSummary", "video-jpea|model:" + imageWorldModel->config().id + "|embeddingDim:" + std::to_string(embedding.size())}, {"rawLocation", cameraDevice}, {"metadata", metadata}}));
            }
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(result.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(e.what());
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/speech/analyze", [&speakIO, &emotionSystem](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("audioBase64")) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("Missing audioBase64");
                cb(resp);
                return;
            }
            std::string b64 = (*json)["audioBase64"].asString();
            if (b64.rfind("data:", 0) == 0) {
                auto pos = b64.find("base64,");
                if (pos != std::string::npos) b64 = b64.substr(pos + 7);
            }
            thread_local std::vector<uint8_t> audioBuf;
            base64DecodeTo(b64, audioBuf);
            auto result = speakIO.analyzeWavBytes(audioBuf);

            std::string sessionId = json->isMember("sessionId") ? (*json)["sessionId"].asString() : "";
            std::string spokenText = result.isMember("text") ? result["text"].asString() : "";
            int turnNumber = 1;
            if (json->isMember("turnNumber"))
                turnNumber = (*json)["turnNumber"].asInt();
            if (!sessionId.empty() && !spokenText.empty()) {
                auto state = emotionSystem.processMessage(sessionId, spokenText, turnNumber);
                Json::Value emo(Json::objectValue);
                emo["valence"] = state.current.valence;
                emo["arousal"] = state.current.arousal;
                emo["dominance"] = state.current.dominance;
                emo["trust"] = state.current.trust;
                emo["joy"] = state.current.joy;
                emo["fear"] = state.current.fear;
                emo["anger"] = state.current.anger;
                emo["surprise"] = state.current.surprise;
                result["emotion"] = emo;
            }

            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(result.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(e.what());
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/speech/synthesize", [&speakIO](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("text")) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("Missing text");
                cb(resp);
                return;
            }
            std::string text = (*json)["text"].asString();
            int sampleRate = json->isMember("sampleRate") ? (*json)["sampleRate"].asInt() : 16000;
            float speed = json->isMember("speed") ? (*json)["speed"].asFloat() : 1.0f;
            float pitch = json->isMember("pitch") ? (*json)["pitch"].asFloat() : 1.0f;
            auto result = speakIO.synthesizeText(text, sampleRate, speed, pitch);
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(result.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(e.what());
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/speech/ingest", [&speakIO, &contextService, &v51Runtime, &worldModel, &emotionSystem, shouldRunV51, buildV51Request](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("audioBase64")) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("Missing audioBase64");
                cb(resp);
                return;
            }
            std::string b64 = (*json)["audioBase64"].asString();
            if (b64.rfind("data:", 0) == 0) {
                auto pos = b64.find("base64,");
                if (pos != std::string::npos) b64 = b64.substr(pos + 7);
            }
            thread_local std::vector<uint8_t> audioBuf;
            base64DecodeTo(b64, audioBuf);
            auto speech = speakIO.analyzeWavBytes(audioBuf);

            std::string sessionId;
            if (json->isMember("sessionId")) sessionId = (*json)["sessionId"].asString();
            if (sessionId.empty()) sessionId = ContextService::generateSessionId();
            std::string mode = "auto";
            if (json->isMember("mode")) mode = (*json)["mode"].asString();

            std::string text = speech.isMember("text") ? speech["text"].asString() : "";
            std::string stage05 = speech.isMember("stage05") ? speech["stage05"].asString() : "";
            std::string combined = text;
            if (!stage05.empty()) {
                if (!combined.empty()) combined += "\n";
                combined += stage05;
            }
            int turnNumber = 1;
            if (json->isMember("turnNumber")) turnNumber = (*json)["turnNumber"].asInt();
            if (!text.empty())
                emotionSystem.processMessage(sessionId, text, turnNumber);
            auto context = contextService.ingest(sessionId, combined, mode);

            Json::Value out;
            out["ok"] = true;
            out["speech"] = speech;
            out["context"] = context;
            out["sessionId"] = sessionId;
            auto worldResult = worldModel.ingestEvidence(nlohmann::json{{"sessionId", sessionId},
                                                                        {"modality", "speech"},
                                                                        {"text", text},
                                                                        {"graphSummary", stage05},
                                                                        {"metadata", nlohmann::json{{"source", "speech/ingest"},
                                                                                                    {"mode", mode},
                                                                                                    {"combined", combined}}}});
            out["worldModel"] = nlohmannToJsonCpp(worldResult);
            if (shouldRunV51(json.get())) {
                Json::Value reqV51 = buildV51Request(json.get(), sessionId, combined, std::string(), text);
                out["v51"] = v51Runtime.process(reqV51);
            } else {
                out["v51"] = Json::Value(Json::nullValue);
            }
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(out.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(e.what());
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/v51/process", [&v51Runtime](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                Json::Value out;
                out["ok"] = false;
                out["error"] = "Missing JSON body";
                resp->setBody(out.toStyledString());
                cb(resp);
                return;
            }
            auto result = v51Runtime.process(*json);
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(result.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            Json::Value out;
            out["ok"] = false;
            out["error"] = "v51 process exception";
            out["message"] = e.what();
            resp->setBody(out.toStyledString());
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/v51/learn", [&v51Runtime](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            auto json = req->getJsonObject();
            if (!json) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setContentTypeString("application/json");
                Json::Value out;
                out["ok"] = false;
                out["error"] = "Missing JSON body";
                resp->setBody(out.toStyledString());
                cb(resp);
                return;
            }
            auto result = v51Runtime.learn(*json);
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(result.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            Json::Value out;
            out["ok"] = false;
            out["error"] = "v51 learn exception";
            out["message"] = e.what();
            resp->setBody(out.toStyledString());
        }
        cb(resp); }, {drogon::Post});

    drogon::app().registerHandler("/v51/status", [&v51Runtime](const drogon::HttpRequestPtr &req, std::function<void(const drogon::HttpResponsePtr &)> &&cb)
                                  {
        auto resp = drogon::HttpResponse::newHttpResponse();
        try {
            const std::string sessionId = req->getParameter("sessionId");
            auto result = v51Runtime.status(sessionId);
            resp->setStatusCode(drogon::k200OK);
            resp->setContentTypeString("application/json");
            resp->setBody(result.toStyledString());
        } catch (const std::exception &e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeString("application/json");
            Json::Value out;
            out["ok"] = false;
            out["error"] = "v51 status exception";
            out["message"] = e.what();
            resp->setBody(out.toStyledString());
        }
        cb(resp); }, {drogon::Get});

    // Emotion system endpoints (four-stage vocabulary weight table)
    auto makeError = [](const std::string& msg) {
        Json::Value out;
        out["ok"] = false;
        out["error"] = msg;
        return out;
    };

    auto emotionStateToJson = [](const phoenix::emotion::EmotionState& s) {
        Json::Value out;
        out["sessionId"] = s.sessionId;
        out["turnNumber"] = static_cast<Json::Int64>(s.turnNumber);
        out["timestampMs"] = static_cast<Json::Int64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                s.timestamp.time_since_epoch()).count());
        out["current"]["valence"] = s.current.valence;
        out["current"]["arousal"] = s.current.arousal;
        out["current"]["dominance"] = s.current.dominance;
        out["current"]["trust"] = s.current.trust;
        out["current"]["joy"] = s.current.joy;
        out["current"]["fear"] = s.current.fear;
        out["current"]["anger"] = s.current.anger;
        out["current"]["surprise"] = s.current.surprise;
        out["baseline"] = out["current"];
        out["stability"] = s.stability();
        out["intensity"] = s.intensity();
        return out;
    };

    drogon::app().registerHandler("/emotion/observe",
                                  [&emotionSystem, &makeError, &emotionStateToJson](
                                      const drogon::HttpRequestPtr& req,
                                      std::function<void(const drogon::HttpResponsePtr&)>&& cb) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeString("application/json");
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("sessionId") || !json->isMember("text")) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody(makeError("sessionId and text required").toStyledString());
                cb(resp);
                return;
            }
            std::string sessionId = (*json)["sessionId"].asString();
            std::string text = (*json)["text"].asString();
            int turnNumber = json->isMember("turnNumber") ? (*json)["turnNumber"].asInt() : 1;
            auto state = emotionSystem.processMessage(sessionId, text, turnNumber);
            Json::Value out;
            out["ok"] = true;
            out["emotionState"] = emotionStateToJson(state);
            resp->setStatusCode(drogon::k200OK);
            resp->setBody(out.toStyledString());
        } catch (const std::exception& e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(makeError(e.what()).toStyledString());
        }
        cb(resp);
    }, {drogon::Post});

    drogon::app().registerHandler("/emotion/bias",
                                  [&emotionSystem, &makeError, &emotionStateToJson](
                                      const drogon::HttpRequestPtr& req,
                                      std::function<void(const drogon::HttpResponsePtr&)>&& cb) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeString("application/json");
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("sessionId") || !json->isMember("text")) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody(makeError("sessionId and text required").toStyledString());
                cb(resp);
                return;
            }
            std::string sessionId = (*json)["sessionId"].asString();
            std::string text = (*json)["text"].asString();
            int turnNumber = json->isMember("turnNumber") ? (*json)["turnNumber"].asInt() : 1;
            auto toks = emotionTokenize(text);
            auto logitBias = emotionSystem.getVocabLogitBias(sessionId, toks, turnNumber);
            auto modulation = emotionSystem.getVocabPromptModulation(sessionId, toks, turnNumber);
            auto emotionCtx = emotionSystem.getEmotionContext(sessionId);
            Json::Value out;
            out["ok"] = true;
            out["logitBias"] = nlohmannToJsonCpp(logitBias);
            out["promptModulation"] = modulation;
            out["emotionContext"] = emotionCtx;
            auto stateOpt = emotionSystem.getEmotionState(sessionId);
            if (stateOpt) out["emotionState"] = emotionStateToJson(*stateOpt);
            resp->setStatusCode(drogon::k200OK);
            resp->setBody(out.toStyledString());
        } catch (const std::exception& e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(makeError(e.what()).toStyledString());
        }
        cb(resp);
    }, {drogon::Post});

    drogon::app().registerHandler("/emotion/feedback",
                                  [&emotionSystem, &makeError](
                                      const drogon::HttpRequestPtr& req,
                                      std::function<void(const drogon::HttpResponsePtr&)>&& cb) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeString("application/json");
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("sessionId") || !json->isMember("prompt") ||
                !json->isMember("reply") || !json->isMember("reward")) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody(makeError("sessionId, prompt, reply and reward required").toStyledString());
                cb(resp);
                return;
            }
            std::string sessionId = (*json)["sessionId"].asString();
            std::string prompt = (*json)["prompt"].asString();
            std::string reply = (*json)["reply"].asString();
            float reward = static_cast<float>((*json)["reward"].asDouble());
            int turnNumber = json->isMember("turnNumber") ? (*json)["turnNumber"].asInt() : 1;
            auto promptToks = emotionTokenize(prompt);
            auto replyToks = emotionTokenize(reply);
            emotionSystem.updateVocabFromResponse(sessionId, promptToks, replyToks, reward, turnNumber);
            Json::Value out;
            out["ok"] = true;
            resp->setStatusCode(drogon::k200OK);
            resp->setBody(out.toStyledString());
        } catch (const std::exception& e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(makeError(e.what()).toStyledString());
        }
        cb(resp);
    }, {drogon::Post});

    drogon::app().registerHandler("/emotion/state",
                                  [&emotionSystem, &makeError, &emotionStateToJson](
                                      const drogon::HttpRequestPtr& req,
                                      std::function<void(const drogon::HttpResponsePtr&)>&& cb) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeString("application/json");
        try {
            std::string sessionId = req->getParameter("sessionId");
            if (sessionId.empty()) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody(makeError("sessionId required").toStyledString());
                cb(resp);
                return;
            }
            auto stateOpt = emotionSystem.getEmotionState(sessionId);
            if (!stateOpt) {
                resp->setStatusCode(drogon::k404NotFound);
                resp->setBody(makeError("session not found").toStyledString());
                cb(resp);
                return;
            }
            Json::Value out;
            out["ok"] = true;
            out["emotionState"] = emotionStateToJson(*stateOpt);
            resp->setStatusCode(drogon::k200OK);
            resp->setBody(out.toStyledString());
        } catch (const std::exception& e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(makeError(e.what()).toStyledString());
        }
        cb(resp);
    }, {drogon::Get});

    drogon::app().registerHandler("/mechanical_mind/analyze",
                                  [&mechanicalMindFilter, &emotionSystem, &makeError, &emotionStateToJson](
                                      const drogon::HttpRequestPtr& req,
                                      std::function<void(const drogon::HttpResponsePtr&)>&& cb) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setContentTypeString("application/json");
        try {
            auto json = req->getJsonObject();
            if (!json || !json->isMember("text")) {
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody(makeError("text required").toStyledString());
                cb(resp);
                return;
            }
            std::string text = (*json)["text"].asString();
            std::string sessionId = json->isMember("sessionId") ? (*json)["sessionId"].asString() : "";
            bool force = json->isMember("force") ? (*json)["force"].asBool() : false;
            bool updateEmotion = json->isMember("updateEmotion") ? (*json)["updateEmotion"].asBool() : false;
            int turnNumber = json->isMember("turnNumber") ? (*json)["turnNumber"].asInt() : 1;

            phoenix::emotion::EmotionTensor current;
            phoenix::emotion::EmotionTensor baseline;
            if (!sessionId.empty()) {
                if (updateEmotion && !text.empty()) {
                    emotionSystem.processMessage(sessionId, text, turnNumber);
                }
                auto stateOpt = emotionSystem.getEmotionState(sessionId);
                if (stateOpt) {
                    current = stateOpt->current;
                    baseline = stateOpt->baseline;
                }
            }

            mechanical_mind::Analysis analysis;
            if (sessionId.empty()) {
                analysis = mechanicalMindFilter.analyzeAndSanitize(text, force);
            } else {
                analysis = mechanicalMindFilter.analyzeAndSanitize(text, current, baseline, force);
            }

            Json::Value out;
            out["ok"] = true;
            out["analysis"] = nlohmannToJsonCpp(analysis.toJson());
            out["text"] = text;
            out["sanitized"] = analysis.sanitized;
            out["triggered"] = analysis.triggered;
            if (!sessionId.empty()) {
                auto stateOpt2 = emotionSystem.getEmotionState(sessionId);
                if (stateOpt2) out["emotionState"] = emotionStateToJson(*stateOpt2);
            }
            resp->setStatusCode(drogon::k200OK);
            resp->setBody(out.toStyledString());
        } catch (const std::exception& e) {
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setBody(makeError(e.what()).toStyledString());
        }
        cb(resp);
    }, {drogon::Post});

    drogon::app().addListener(host, port);

    std::cout << "[frontend_server] Listening on http://" << host << ":" << port << std::endl;
    std::cout << "[frontend_server] webRoot: " << webRoot << std::endl;
}