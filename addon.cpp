/* addon.cpp - Addon system implementation
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

#include "addon.hpp"

#include "addons/builtin_registry.hpp"
#include "util/module_resource.hpp"

#include <algorithm>
#include <filesystem>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace addon {

static std::mutex gAddonHandlerMu;
static AddonOnlineLookupHandler gAddonOnlineLookupHandler;
static AddonComputerShellHandler gAddonComputerShellHandler;

void setAddonOnlineLookupHandler(AddonOnlineLookupHandler handler) {
	std::lock_guard<std::mutex> lock(gAddonHandlerMu);
	gAddonOnlineLookupHandler = std::move(handler);
}

void clearAddonOnlineLookupHandler() {
	std::lock_guard<std::mutex> lock(gAddonHandlerMu);
	gAddonOnlineLookupHandler = nullptr;
}

bool invokeAddonOnlineLookup(const json &input, const json &options, json &out) {
	AddonOnlineLookupHandler handler;
	{
		std::lock_guard<std::mutex> lock(gAddonHandlerMu);
		handler = gAddonOnlineLookupHandler;
	}
	if (!handler) return false;
	try {
		out = handler(input, options);
		return true;
	} catch (...) {
		out = json::object();
		return false;
	}
}

void setAddonComputerShellHandler(AddonComputerShellHandler handler) {
	gAddonComputerShellHandler = std::move(handler);
}

void clearAddonComputerShellHandler() {
	gAddonComputerShellHandler = nullptr;
}

bool invokeAddonComputerShell(const json &request, const json &options, json &out) {
	if (!gAddonComputerShellHandler) return false;
	try {
		out = gAddonComputerShellHandler(request, options);
		return true;
	} catch (...) {
		out = json::object();
		return false;
	}
}

void AddonManager::registerAddon(std::shared_ptr<Addon> addon) {
	if (!addon) return;
	std::lock_guard<std::mutex> lock(mu_);
	addRecord(addon, "builtin", "", nullptr, nullptr);
}

bool AddonManager::addBuiltin(const std::string &type, const std::string &name, std::string *error) {
	std::string key = name.empty() ? type : name;
	if (key.empty()) {
		if (error) *error = "name required";
		return false;
	}
	auto addon = builtins::createBuiltinAddon(type, key, error);
	if (!addon) return false;
	std::lock_guard<std::mutex> lock(mu_);
	return addRecord(addon, "builtin", "", nullptr, error);
}

bool AddonManager::loadLibrary(const std::string &path, std::string *error) {
	if (path.empty()) {
		if (error) *error = "path required";
		return false;
	}
#ifdef _WIN32
	HMODULE lib = LoadLibraryA(path.c_str());
	if (!lib) {
		if (error) *error = "failed to load library";
		return false;
	}
	auto apiFn = (int (*)())GetProcAddress(lib, "addon_api_version");
	auto createFn = (Addon *(*)())GetProcAddress(lib, "addon_create_v1");
	auto destroyFn = (void (*)(Addon *))GetProcAddress(lib, "addon_destroy_v1");
#else
	void *lib = dlopen(path.c_str(), RTLD_NOW);
	if (!lib) {
		if (error) *error = dlerror();
		return false;
	}
	auto apiFn = (int (*)())dlsym(lib, "addon_api_version");
	auto createFn = (Addon *(*)())dlsym(lib, "addon_create_v1");
	auto destroyFn = (void (*)(Addon *))dlsym(lib, "addon_destroy_v1");
#endif
	if (!createFn || !destroyFn) {
		if (error) *error = "missing addon_create_v1/addon_destroy_v1";
#ifdef _WIN32
		FreeLibrary(lib);
#else
		dlclose(lib);
#endif
		return false;
	}
	int apiVersion = apiFn ? apiFn() : 1;
	if (apiVersion != 1 && apiVersion != 2) {
		if (error) *error = "unsupported addon api version";
#ifdef _WIN32
		FreeLibrary(lib);
#else
		dlclose(lib);
#endif
		return false;
	}
	Addon *raw = createFn();
	if (!raw) {
		if (error) *error = "addon_create_v1 returned null";
#ifdef _WIN32
		FreeLibrary(lib);
#else
		dlclose(lib);
#endif
		return false;
	}
	std::shared_ptr<Addon> addon(raw, [destroyFn](Addon *p) { destroyFn(p); });
	std::lock_guard<std::mutex> lock(mu_);
	if (!addRecord(addon, "library", path, lib, error, apiVersion)) {
		addon.reset();
#ifdef _WIN32
		FreeLibrary(lib);
#else
		dlclose(lib);
#endif
		return false;
	}
	return true;
}

bool AddonManager::removeAddon(const std::string &name, std::string *error) {
	if (name.empty()) {
		if (error) *error = "name required";
		return false;
	}
	std::lock_guard<std::mutex> lock(mu_);
	auto it = addonIndex_.find(name);
	if (it == addonIndex_.end()) {
		if (error) *error = "addon not found";
		return false;
	}
	const AddonRef ref = it->second;
	auto &store = ref.mounted ? mountedAddons_ : builtinAddons_;
	if (ref.index < store.size()) {
		unloadRecord(store[ref.index], ref.mounted);
		store.erase(store.begin() + (std::ptrdiff_t)ref.index);
	}
	rebuildIndex();
	phoenix::util::addonCatalogStore().erase(name);
	return true;
}

json AddonManager::listAddons() const {
	std::lock_guard<std::mutex> lock(mu_);
	json out = json::array();
	auto append = [&](const std::vector<AddonRecord> &records) {
		for (const auto &rec : records) {
			out.push_back(json{{"name", rec.name},
			                   {"type", rec.type},
			                   {"source", rec.source},
			                   {"path", rec.path},
			                   {"apiVersion", rec.apiVersion},
			                   {"selfOffer", rec.apiVersion >= 2}});
		}
	};
	append(builtinAddons_);
	append(mountedAddons_);
	return out;
}

AddonResult AddonManager::run(const std::string &text, const json &payload) const {
	std::vector<std::shared_ptr<Addon>> snap;
	{
		std::lock_guard<std::mutex> lock(mu_);
		auto take = [&](const std::vector<AddonRecord> &records) {
			for (const auto &rec : records) {
				if (rec.addon) snap.push_back(rec.addon);
			}
		};
		take(builtinAddons_);
		take(mountedAddons_);
	}
	AddonResult combined;
	for (const auto &addon : snap) {
		auto res = addon->handle(text, payload);
		if (res.handled) return res;
		if (!res.extraTokens.empty()) {
			combined.extraTokens.insert(combined.extraTokens.end(), res.extraTokens.begin(), res.extraTokens.end());
		}
	}
	return combined;
}

std::vector<AddonOffer> AddonManager::collectOffers(const json &situation,
                                                    float minScore,
                                                    size_t maxOffers) const {
	struct Snap {
		std::shared_ptr<Addon> addon;
		std::string name;
		std::string type;
		int apiVersion{2};
	};
	std::vector<Snap> snap;
	{
		std::lock_guard<std::mutex> lock(mu_);
		auto take = [&](const std::vector<AddonRecord> &records) {
			for (const auto &rec : records) {
				if (!rec.addon) continue;
				snap.push_back({rec.addon, rec.name, rec.type, rec.apiVersion});
			}
		};
		take(builtinAddons_);
		take(mountedAddons_);
	}
	std::vector<AddonOffer> offers;
	for (const auto &s : snap) {
		if (s.apiVersion < 2) continue;
		float score = 0.f;
		try {
			score = s.addon->consider(situation);
		} catch (...) {
			continue;
		}
		if (score < minScore) continue;
		AddonOffer offer;
		offer.score = score;
		offer.name = s.name;
		offer.type = s.type;
		try {
			offer.result = s.addon->contribute(situation);
		} catch (...) {
			continue;
		}
		sealAddonResultUnits(offer.result);
		if (offer.result.meta.is_object() &&
		    offer.result.meta.contains("reason") &&
		    offer.result.meta["reason"].is_string())
			offer.reason = offer.result.meta["reason"].get<std::string>();
		if (offer.result.handled) offers.push_back(std::move(offer));
	}
	std::sort(offers.begin(), offers.end(),
	          [](const AddonOffer &a, const AddonOffer &b) {
		          return a.score > b.score;
	          });
	if (maxOffers > 0 && offers.size() > maxOffers) offers.resize(maxOffers);
	return offers;
}

int AddonManager::scanAutoloadDir(const std::string &dir, json *report) {
	namespace fs = std::filesystem;
	if (dir.empty()) return 0;
	std::error_code ec;
	if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return 0;
	std::vector<std::string> already;
	{
		std::lock_guard<std::mutex> lock(mu_);
		for (const auto &rec : mountedAddons_) {
			if (!rec.path.empty()) already.push_back(rec.path);
		}
	}
	int loaded = 0;
	for (const auto &entry : fs::directory_iterator(dir, ec)) {
		if (ec) break;
		if (!entry.is_regular_file(ec)) continue;
		const auto ext = entry.path().extension().string();
		if (ext != ".so" && ext != ".dll") continue;
		const std::string path = entry.path().string();
		bool skip = false;
		for (const auto &a : already) {
			if (a == path) {
				skip = true;
				break;
			}
		}
		if (skip) continue;
		std::string error;
		if (loadLibrary(path, &error)) {
			++loaded;
			already.push_back(path);
		} else if (report) {
			(*report)[path] = error.empty() ? std::string("load failed") : error;
		}
	}
	return loaded;
}

bool AddonManager::addRecord(const std::shared_ptr<Addon> &addon,
						 const std::string &source,
						 const std::string &path,
						 void *libHandle,
						 std::string *error,
						 int apiVersion) {
	if (!addon) {
		if (error) *error = "addon is null";
		return false;
	}
	std::string key = addon->name();
	if (key.empty()) {
		if (error) *error = "addon name required";
		return false;
	}
	if (addonIndex_.count(key)) {
		if (error) *error = "addon already exists";
		return false;
	}
	auto &store = pickStore(source);
	AddonRecord rec;
	rec.addon = addon;
	rec.name = key;
	rec.type = addon->type();
	rec.source = source;
	rec.path = path;
	rec.libHandle = libHandle;
	rec.apiVersion = apiVersion < 1 ? 1 : apiVersion;
	phoenix::util::addonCatalogStore().put(key, json{{"id", key},
	                                                 {"name", key},
	                                                 {"type", rec.type},
	                                                 {"source", rec.source},
	                                                 {"path", rec.path},
	                                                 {"apiVersion", rec.apiVersion}});
	store.push_back(std::move(rec));
	rebuildIndex();
	return true;
}

std::vector<AddonManager::AddonRecord> &AddonManager::pickStore(const std::string &source) {
	if (source == "library" || source == "mounted") return mountedAddons_;
	return builtinAddons_;
}

const AddonManager::AddonRecord *AddonManager::findRecord(const std::string &name) const {
	auto it = addonIndex_.find(name);
	if (it == addonIndex_.end()) return nullptr;
	const auto &ref = it->second;
	const auto &store = ref.mounted ? mountedAddons_ : builtinAddons_;
	if (ref.index >= store.size()) return nullptr;
	return &store[ref.index];
}

AddonManager::AddonRecord *AddonManager::findRecordMutable(const std::string &name) {
	auto it = addonIndex_.find(name);
	if (it == addonIndex_.end()) return nullptr;
	const auto &ref = it->second;
	auto &store = ref.mounted ? mountedAddons_ : builtinAddons_;
	if (ref.index >= store.size()) return nullptr;
	return &store[ref.index];
}

void AddonManager::rebuildIndex() {
	addonIndex_.clear();
	for (size_t i = 0; i < builtinAddons_.size(); i++) {
		addonIndex_[builtinAddons_[i].name] = AddonRef{false, i};
	}
	for (size_t i = 0; i < mountedAddons_.size(); i++) {
		addonIndex_[mountedAddons_[i].name] = AddonRef{true, i};
	}
}

void AddonManager::unloadRecord(AddonRecord &rec, bool mounted) {
	rec.addon.reset();
	if (mounted && rec.libHandle) {
#ifdef _WIN32
		FreeLibrary((HMODULE)rec.libHandle);
#else
		dlclose(rec.libHandle);
#endif
		rec.libHandle = nullptr;
	}
}

} // namespace addon
