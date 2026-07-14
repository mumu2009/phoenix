/* addon.hpp - Plugin/addon system for 079 Project
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

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

#include <nlohmann/json.hpp>

namespace addon {

using json = nlohmann::json;

/* Result structure returned by addon handlers.
   Contains the reply, extra tokens, and metadata. */
struct AddonResult {
	bool handled{false};              /* Whether the addon handled the request */
	std::string reply;                /* Reply text from the addon */
	std::vector<std::string> extraTokens; /* Additional tokens to inject */
	json meta = json::object();       /* Metadata about the handling */
};

/* Abstract base class for addons/plugins.
   Addons extend the system's functionality by handling specific requests. */
class Addon {
public:
	virtual ~Addon() = default;
	virtual std::string name() const = 0; /* Addon name */
	virtual std::string type() const = 0; /* Addon type (e.g., "math", "search") */
	virtual AddonResult handle(const std::string &text, const json &payload) = 0; /* Handle a request */
};

/* Manager for addon registration, loading, and execution.
   Supports both built-in addons and dynamically loaded libraries. */
class AddonManager {
public:
	void registerAddon(std::shared_ptr<Addon> addon); /* Register an addon instance */
	bool addBuiltin(const std::string &type, const std::string &name, std::string *error = nullptr); /* Add a built-in addon */
	bool loadLibrary(const std::string &path, std::string *error = nullptr); /* Load addon from dynamic library */
	bool removeAddon(const std::string &name, std::string *error = nullptr); /* Remove an addon */
	json listAddons() const; /* List all registered addons */
	AddonResult run(const std::string &text, const json &payload) const; /* Run addons on a request */

private:
	/* Internal record for a registered addon */
	struct AddonRecord {
		std::shared_ptr<Addon> addon; /* Addon instance */
		std::string name;              /* Addon name */
		std::string type;              /* Addon type */
		std::string source;            /* Source (builtin or library) */
		std::string path;              /* Library path (if loaded from library) */
		void *libHandle{nullptr};      /* Dynamic library handle */
	};

	/* Reference to an addon in the index */
	struct AddonRef {
		bool mounted{false};           /* Whether addon is mounted */
		size_t index{0};              /* Index in the store */
	};

	bool addRecord(const std::shared_ptr<Addon> &addon,
				   const std::string &source,
				   const std::string &path,
				   void *libHandle,
				   std::string *error); /* Add an addon record */
	std::vector<AddonRecord> &pickStore(const std::string &source); /* Get store by source */
	const AddonRecord *findRecord(const std::string &name) const; /* Find addon record by name */
	AddonRecord *findRecordMutable(const std::string &name); /* Find mutable addon record */
	void rebuildIndex(); /* Rebuild the addon index */
	void unloadRecord(AddonRecord &rec, bool mounted); /* Unload an addon record */

	mutable std::mutex mu_;           /* Mutex for thread safety */
	std::vector<AddonRecord> builtinAddons_; /* Built-in addons */
	std::vector<AddonRecord> mountedAddons_; /* Dynamically loaded addons */
	std::unordered_map<std::string, AddonRef> addonIndex_; /* Name to addon mapping */
};

using AddonOnlineLookupHandler = std::function<json(const json &input, const json &options)>; /* Online lookup handler type */
using AddonComputerShellHandler = std::function<json(const json &request, const json &options)>; /* Computer shell handler type */

void setAddonOnlineLookupHandler(AddonOnlineLookupHandler handler); /* Set online lookup handler */
void clearAddonOnlineLookupHandler(); /* Clear online lookup handler */
bool invokeAddonOnlineLookup(const json &input, const json &options, json &out); /* Invoke online lookup */
void setAddonComputerShellHandler(AddonComputerShellHandler handler); /* Set computer shell handler */
void clearAddonComputerShellHandler(); /* Clear computer shell handler */
bool invokeAddonComputerShell(const json &request, const json &options, json &out); /* Invoke computer shell */

std::shared_ptr<AddonManager> createDefaultAddons(const std::string &builtinSelection = std::string()); /* Create default addon manager */

} // namespace addon
