/* module_resource.hpp - Shared module-internal resource registry + dual CRUD
   Copyright (C) 2026 079 Project

   External HTTP and in-process plugin/util CRUD share one registry.
   Unregistered resources are refused. ACL is checked before handlers run. */

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../module_mount.hpp"
#include "../plugin_system.hpp"

namespace phoenix {
namespace util {

using json = nlohmann::json;

enum class CrudOp { List, Get, Create, Update, Delete };

enum class CrudChannel { External, Internal };

inline const char *crudOpName(CrudOp op) {
    switch (op) {
    case CrudOp::List:
        return "list";
    case CrudOp::Get:
        return "get";
    case CrudOp::Create:
        return "create";
    case CrudOp::Update:
        return "update";
    case CrudOp::Delete:
        return "delete";
    }
    return "unknown";
}

inline bool crudOpWrites(CrudOp op) {
    return op == CrudOp::Create || op == CrudOp::Update || op == CrudOp::Delete;
}

inline plugin::PluginCapability capabilityForOp(CrudOp op) {
    if (op == CrudOp::Delete)
        return plugin::PluginCapability::DELETE_DATA;
    if (crudOpWrites(op))
        return plugin::PluginCapability::WRITE_DATA;
    return plugin::PluginCapability::READ_DATA;
}

struct ResourceAcl {
    bool allowExternal{true};
    bool allowInternal{true};
    bool allowExternalWrite{false};
    bool allowInternalWrite{false};
    std::vector<plugin::PluginCapability> requiredCaps;
    std::vector<std::string> allowedActors;
};

struct CrudCall {
    CrudOp op{CrudOp::Get};
    CrudChannel channel{CrudChannel::Internal};
    std::string moduleId;
    std::string type;
    std::string id;
    json body = json::object();
    std::string actor;
    std::vector<plugin::PluginCapability> actorCaps;
    std::string bearerToken;
};

struct CrudReply {
    bool ok{false};
    int httpStatus{404};
    std::string error;
    json data = json::object();
};

using ResourceFn = std::function<CrudReply(const CrudCall &)>;

struct ResourceSpec {
    std::string moduleId;
    std::string type;
    std::string description;
    ResourceAcl acl;
    ResourceFn handler;
};

inline std::string resourceKey(const std::string &moduleId, const std::string &type) {
    return moduleId + "/" + type;
}

inline bool actorHasCap(const std::vector<plugin::PluginCapability> &caps,
                        plugin::PluginCapability need) {
    for (auto c : caps) {
        if (c == need || c == plugin::PluginCapability::FULL_ACCESS)
            return true;
    }
    return false;
}

inline bool actorHasAll(const std::vector<plugin::PluginCapability> &caps,
                        const std::vector<plugin::PluginCapability> &need) {
    for (auto n : need) {
        if (!actorHasCap(caps, n))
            return false;
    }
    return true;
}

inline bool actorAllowed(const ResourceAcl &acl, const std::string &actor) {
    if (acl.allowedActors.empty())
        return true;
    for (const auto &a : acl.allowedActors) {
        if (a == actor)
            return true;
    }
    return false;
}

struct SessionMetaRecord {
    std::string id;
    std::string label;
    std::int64_t createdAt{0};
    std::int64_t lastSeen{0};
};

class SessionMetaStore {
public:
    void upsert(const SessionMetaRecord &rec) {
        if (rec.id.empty())
            return;
        std::lock_guard<std::mutex> lock(mu_);
        items_[rec.id] = rec;
    }
    void erase(const std::string &id) {
        std::lock_guard<std::mutex> lock(mu_);
        items_.erase(id);
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        items_.clear();
    }
    json getJson(const std::string &id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = items_.find(id);
        if (it == items_.end())
            return json();
        return toJson(it->second);
    }
    json listJson() const {
        std::lock_guard<std::mutex> lock(mu_);
        json arr = json::array();
        for (const auto &p : items_)
            arr.push_back(toJson(p.second));
        return arr;
    }

private:
    static json toJson(const SessionMetaRecord &r) {
        return json{{"id", r.id},
                    {"label", r.label},
                    {"createdAt", r.createdAt},
                    {"lastSeen", r.lastSeen}};
    }
    mutable std::mutex mu_;
    std::map<std::string, SessionMetaRecord> items_;
};

class CatalogStore {
public:
    void put(const std::string &id, const json &row) {
        if (id.empty())
            return;
        std::lock_guard<std::mutex> lock(mu_);
        items_[id] = row.is_object() ? row : json::object();
        items_[id]["id"] = id;
    }
    void erase(const std::string &id) {
        std::lock_guard<std::mutex> lock(mu_);
        items_.erase(id);
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        items_.clear();
    }
    void replaceAll(const json &arr) {
        std::lock_guard<std::mutex> lock(mu_);
        items_.clear();
        if (!arr.is_array())
            return;
        for (const auto &row : arr) {
            if (!row.is_object())
                continue;
            std::string id = row.value("id", row.value("name", std::string()));
            if (id.empty())
                continue;
            items_[id] = row;
            items_[id]["id"] = id;
        }
    }
    json getJson(const std::string &id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = items_.find(id);
        if (it == items_.end())
            return json();
        return it->second;
    }
    json listJson() const {
        std::lock_guard<std::mutex> lock(mu_);
        json arr = json::array();
        for (const auto &p : items_)
            arr.push_back(p.second);
        return arr;
    }

private:
    mutable std::mutex mu_;
    std::map<std::string, json> items_;
};

inline SessionMetaStore &sessionMetaStore() {
    static SessionMetaStore store;
    return store;
}

inline CatalogStore &pluginCatalogStore() {
    static CatalogStore store;
    return store;
}

inline CatalogStore &addonCatalogStore() {
    static CatalogStore store;
    return store;
}

inline json &publicConfigStore() {
    static json cfg = json::object();
    return cfg;
}

inline std::mutex &publicConfigMutex() {
    static std::mutex mu;
    return mu;
}

inline void setPublicConfig(const json &cfg) {
    std::lock_guard<std::mutex> lock(publicConfigMutex());
    publicConfigStore() = cfg.is_object() ? cfg : json::object();
}

inline json getPublicConfig() {
    std::lock_guard<std::mutex> lock(publicConfigMutex());
    return publicConfigStore();
}

struct CrudHttpConfig {
    bool enabled{true};
    std::string token;
    bool acceptLocalToken{true};
};

inline CrudHttpConfig &crudHttpConfigMut() {
    static CrudHttpConfig cfg;
    return cfg;
}

inline std::mutex &crudHttpConfigMutex() {
    static std::mutex mu;
    return mu;
}

inline void setCrudHttpConfig(const CrudHttpConfig &cfg) {
    std::lock_guard<std::mutex> lock(crudHttpConfigMutex());
    crudHttpConfigMut() = cfg;
}

inline CrudHttpConfig getCrudHttpConfig() {
    std::lock_guard<std::mutex> lock(crudHttpConfigMutex());
    return crudHttpConfigMut();
}

inline bool bearerAuthorized(const std::string &token) {
    CrudHttpConfig cfg = getCrudHttpConfig();
    if (token.empty())
        return false;
    if (cfg.acceptLocalToken && token.rfind("local-", 0) == 0)
        return true;
    if (!cfg.token.empty() && token == cfg.token)
        return true;
    return false;
}

class ModuleResourceRegistry {
public:
    static ModuleResourceRegistry &instance() {
        static ModuleResourceRegistry inst;
        return inst;
    }

    bool registerResource(ResourceSpec spec) {
        if (spec.moduleId.empty() || spec.type.empty() || !spec.handler)
            return false;
        std::lock_guard<std::mutex> lock(mu_);
        specs_[resourceKey(spec.moduleId, spec.type)] = std::move(spec);
        return true;
    }

    bool unregisterResource(const std::string &moduleId, const std::string &type) {
        std::lock_guard<std::mutex> lock(mu_);
        return specs_.erase(resourceKey(moduleId, type)) > 0;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        specs_.clear();
        builtinsOn_ = false;
    }

    bool isRegistered(const std::string &moduleId, const std::string &type) const {
        std::lock_guard<std::mutex> lock(mu_);
        return specs_.find(resourceKey(moduleId, type)) != specs_.end();
    }

    json listSpecs() const {
        std::lock_guard<std::mutex> lock(mu_);
        json arr = json::array();
        for (const auto &p : specs_) {
            arr.push_back(json{{"moduleId", p.second.moduleId},
                               {"type", p.second.type},
                               {"description", p.second.description},
                               {"allowExternal", p.second.acl.allowExternal},
                               {"allowInternal", p.second.acl.allowInternal},
                               {"allowExternalWrite", p.second.acl.allowExternalWrite},
                               {"allowInternalWrite", p.second.acl.allowInternalWrite}});
        }
        return arr;
    }

    CrudReply execute(const CrudCall &call) const {
        ResourceSpec spec;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = specs_.find(resourceKey(call.moduleId, call.type));
            if (it == specs_.end()) {
                CrudReply r;
                r.ok = false;
                r.httpStatus = 404;
                r.error = "unregistered";
                r.data = json{{"moduleId", call.moduleId}, {"type", call.type}};
                return r;
            }
            spec = it->second;
        }

        if (call.channel == CrudChannel::External && !spec.acl.allowExternal)
            return deny(403, "forbidden", "external_disabled");
        if (call.channel == CrudChannel::Internal && !spec.acl.allowInternal)
            return deny(403, "forbidden", "internal_disabled");
        if (crudOpWrites(call.op)) {
            if (call.channel == CrudChannel::External && !spec.acl.allowExternalWrite)
                return deny(405, "method_not_allowed", "external_read_only");
            if (call.channel == CrudChannel::Internal && !spec.acl.allowInternalWrite)
                return deny(405, "method_not_allowed", "internal_read_only");
        }
        if (call.channel == CrudChannel::External && !bearerAuthorized(call.bearerToken))
            return deny(401, "unauthorized", "bad_token");
        if (call.channel == CrudChannel::Internal) {
            if (!actorAllowed(spec.acl, call.actor))
                return deny(403, "forbidden", "actor_denied");
            auto need = spec.acl.requiredCaps;
            need.push_back(capabilityForOp(call.op));
            if (!actorHasAll(call.actorCaps, need))
                return deny(403, "forbidden", "capability_denied");
        }
        if (call.op == CrudOp::Get && call.id.empty())
            return deny(400, "bad_request", "missing_id");
        return spec.handler(call);
    }

    bool builtinsInstalled() const {
        std::lock_guard<std::mutex> lock(mu_);
        return builtinsOn_;
    }
    void markBuiltinsInstalled() {
        std::lock_guard<std::mutex> lock(mu_);
        builtinsOn_ = true;
    }

private:
    static CrudReply deny(int status, const std::string &error, const std::string &reason) {
        CrudReply r;
        r.ok = false;
        r.httpStatus = status;
        r.error = error;
        r.data = json{{"reason", reason}};
        return r;
    }

    mutable std::mutex mu_;
    std::map<std::string, ResourceSpec> specs_;
    bool builtinsOn_{false};
};

inline CrudReply storeListGet(const CatalogStore &store, const CrudCall &call) {
    CrudReply r;
    if (call.op == CrudOp::List) {
        r.ok = true;
        r.httpStatus = 200;
        r.data = json{{"items", store.listJson()}};
        return r;
    }
    json row = store.getJson(call.id);
    if (row.is_null() || row.empty()) {
        r.ok = false;
        r.httpStatus = 404;
        r.error = "not_found";
        return r;
    }
    r.ok = true;
    r.httpStatus = 200;
    r.data = row;
    return r;
}

inline CrudReply handleSessionMeta(const CrudCall &call) {
    auto &store = sessionMetaStore();
    if (call.op == CrudOp::List) {
        CrudReply r;
        r.ok = true;
        r.httpStatus = 200;
        r.data = json{{"items", store.listJson()}};
        return r;
    }
    if (call.op == CrudOp::Get) {
        json row = store.getJson(call.id);
        if (row.is_null() || row.empty())
            return CrudReply{false, 404, "not_found", json::object()};
        CrudReply r;
        r.ok = true;
        r.httpStatus = 200;
        r.data = row;
        return r;
    }
    if (call.op == CrudOp::Create || call.op == CrudOp::Update) {
        SessionMetaRecord rec;
        rec.id = call.id.empty() ? call.body.value("id", std::string()) : call.id;
        rec.label = call.body.value("label", std::string());
        rec.createdAt = call.body.value("createdAt", (std::int64_t)0);
        rec.lastSeen = call.body.value("lastSeen", (std::int64_t)0);
        if (rec.id.empty()) {
            return CrudReply{false, 400, "bad_request", json{{"reason", "missing_id"}}};
        }
        if (rec.createdAt == 0) {
            rec.createdAt = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();
        }
        store.upsert(rec);
        CrudReply r;
        r.ok = true;
        r.httpStatus = call.op == CrudOp::Create ? 201 : 200;
        r.data = store.getJson(rec.id);
        return r;
    }
    if (call.op == CrudOp::Delete) {
        store.erase(call.id);
        return CrudReply{true, 200, "", json{{"deleted", call.id}}};
    }
    return CrudReply{false, 405, "method_not_allowed", json::object()};
}

inline CrudReply handlePluginCatalog(const CrudCall &call) {
    json live = json::array();
    for (const auto &t : plugin::PluginRegistry::instance().getRegisteredTypes()) {
        live.push_back(json{{"id", t}, {"name", t}, {"kind", "factory"}});
    }
    json stored = pluginCatalogStore().listJson();
    if (stored.is_array()) {
        for (const auto &row : stored)
            live.push_back(row);
    }
    if (call.op == CrudOp::List) {
        CrudReply r;
        r.ok = true;
        r.httpStatus = 200;
        r.data = json{{"items", live}};
        return r;
    }
    for (const auto &row : live) {
        if (row.value("id", std::string()) == call.id) {
            CrudReply r;
            r.ok = true;
            r.httpStatus = 200;
            r.data = row;
            return r;
        }
    }
    return CrudReply{false, 404, "not_found", json::object()};
}

inline CrudReply handleAddonMounted(const CrudCall &call) {
    return storeListGet(addonCatalogStore(), call);
}

inline CrudReply handlePublicConfig(const CrudCall &call) {
    json cfg = getPublicConfig();
    if (call.op == CrudOp::List) {
        json items = json::array();
        if (cfg.is_object()) {
            for (auto it = cfg.begin(); it != cfg.end(); ++it)
                items.push_back(json{{"id", it.key()}, {"value", it.value()}});
        }
        CrudReply r;
        r.ok = true;
        r.httpStatus = 200;
        r.data = json{{"items", items}};
        return r;
    }
    if (call.op == CrudOp::Get) {
        if (!cfg.is_object() || !cfg.contains(call.id))
            return CrudReply{false, 404, "not_found", json::object()};
        CrudReply r;
        r.ok = true;
        r.httpStatus = 200;
        r.data = json{{"id", call.id}, {"value", cfg[call.id]}};
        return r;
    }
    return CrudReply{false, 405, "method_not_allowed", json::object()};
}

inline CrudReply handleModuleMountFactories(const CrudCall &call) {
    json presence = module_mount::ModuleRegistry::instance().factoryPresence();
    if (call.op == CrudOp::List) {
        json items = json::array();
        for (auto it = presence.begin(); it != presence.end(); ++it)
            items.push_back(json{{"id", it.key()}, {"registered", it.value()}});
        CrudReply r;
        r.ok = true;
        r.httpStatus = 200;
        r.data = json{{"items", items}};
        return r;
    }
    if (!presence.contains(call.id))
        return CrudReply{false, 404, "not_found", json::object()};
    CrudReply r;
    r.ok = true;
    r.httpStatus = 200;
    r.data = json{{"id", call.id}, {"registered", presence[call.id]}};
    return r;
}

inline void installBuiltinModuleResources() {
    auto &reg = ModuleResourceRegistry::instance();
    if (reg.builtinsInstalled())
        return;

    ResourceAcl readExt;
    readExt.allowExternal = true;
    readExt.allowInternal = true;
    readExt.allowExternalWrite = false;
    readExt.allowInternalWrite = false;

    ResourceAcl sessionAcl = readExt;
    sessionAcl.allowInternalWrite = true;

    reg.registerResource(ResourceSpec{"session", "meta",
                                      "session metadata only (no embeddings)", sessionAcl,
                                      handleSessionMeta});
    reg.registerResource(ResourceSpec{"plugin", "catalog",
                                      "plugin factory + published plugin rows", readExt,
                                      handlePluginCatalog});
    reg.registerResource(ResourceSpec{"addon", "mounted",
                                      "mounted addon inventory", readExt,
                                      handleAddonMounted});
    reg.registerResource(ResourceSpec{"config", "public",
                                      "user-visible config keys", readExt,
                                      handlePublicConfig});
    reg.registerResource(ResourceSpec{"module_mount", "factories",
                                      "which module factories are installed", readExt,
                                      handleModuleMountFactories});
    reg.markBuiltinsInstalled();
}

inline bool parseModuleResourcePath(const std::string &path,
                                    std::string &moduleId,
                                    std::string &type,
                                    std::string &id) {
    moduleId.clear();
    type.clear();
    id.clear();
    const std::string prefix = "/api/modules/";
    if (path.rfind(prefix, 0) != 0)
        return false;
    std::string rest = path.substr(prefix.size());
    auto slash = rest.find('/');
    if (slash == std::string::npos || slash == 0)
        return false;
    moduleId = rest.substr(0, slash);
    rest = rest.substr(slash + 1);
    const std::string mid = "resources/";
    if (rest.rfind(mid, 0) != 0)
        return false;
    rest = rest.substr(mid.size());
    if (rest.empty())
        return false;
    slash = rest.find('/');
    if (slash == std::string::npos) {
        type = rest;
        return !type.empty();
    }
    type = rest.substr(0, slash);
    id = rest.substr(slash + 1);
    return !moduleId.empty() && !type.empty();
}

inline CrudOp crudOpFromHttp(const std::string &method, bool hasId) {
    std::string m = method;
    for (char &c : m) {
        if (c >= 'a' && c <= 'z')
            c = static_cast<char>(c - 'a' + 'A');
    }
    if (m == "GET")
        return hasId ? CrudOp::Get : CrudOp::List;
    if (m == "POST")
        return CrudOp::Create;
    if (m == "PUT" || m == "PATCH")
        return CrudOp::Update;
    if (m == "DELETE")
        return CrudOp::Delete;
    return CrudOp::Get;
}

inline std::string extractBearer(const std::string &authorization) {
    if (authorization.size() >= 7) {
        std::string head = authorization.substr(0, 7);
        for (char &c : head) {
            if (c >= 'A' && c <= 'Z')
                c = static_cast<char>(c - 'A' + 'a');
        }
        if (head == "bearer ")
            return authorization.substr(7);
    }
    return std::string();
}

inline CrudReply handleExternalCrud(const std::string &method,
                                    const std::string &path,
                                    const std::string &authorization,
                                    const json &body) {
    installBuiltinModuleResources();
    CrudCall call;
    call.channel = CrudChannel::External;
    call.bearerToken = extractBearer(authorization);
    call.body = body.is_object() ? body : json::object();
    if (!parseModuleResourcePath(path, call.moduleId, call.type, call.id)) {
        return CrudReply{false, 404, "unregistered", json{{"reason", "bad_path"}}};
    }
    call.op = crudOpFromHttp(method, !call.id.empty());
    if (call.id.empty())
        call.id = call.body.value("id", std::string());
    return ModuleResourceRegistry::instance().execute(call);
}

inline CrudReply handleInternalCrud(const std::string &actor,
                                    const std::vector<plugin::PluginCapability> &caps,
                                    CrudOp op,
                                    const std::string &moduleId,
                                    const std::string &type,
                                    const std::string &id,
                                    const json &body) {
    installBuiltinModuleResources();
    CrudCall call;
    call.op = op;
    call.channel = CrudChannel::Internal;
    call.moduleId = moduleId;
    call.type = type;
    call.id = id;
    call.body = body.is_object() ? body : json::object();
    call.actor = actor;
    call.actorCaps = caps;
    return ModuleResourceRegistry::instance().execute(call);
}

} // namespace util
} // namespace phoenix
