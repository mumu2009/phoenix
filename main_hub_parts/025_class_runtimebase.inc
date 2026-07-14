/* Auto-generated from main.cpp
   Segment: class_RuntimeBase
   Original lines: 3415-3440
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

class RuntimeBase {
public:
  virtual ~RuntimeBase() = default;
  virtual json processInput(const json &payload) = 0;
  virtual json cloneParams() const = 0;
  virtual void setParams(const json &patch) = 0;
  virtual json toSnapshot() = 0;
  virtual void fromSnapshot(const json &snapshot) = 0;
  virtual json ingestDocument(const json &doc) = 0;
  virtual json forgetMemes(const json &payload) = 0;
  virtual json onlineLookup(const std::string &query, const json &options) = 0;
  virtual json onlineLookup(const json &input, const json &options) = 0;
  virtual json getSearchConfig() const = 0;
  virtual json setSearchConfig(const json &patch) = 0;
  virtual std::vector<RobotsDocument>
  collectRobotsDocuments(const json &options) const = 0;
  virtual std::vector<std::string> listRobotsFiles() const = 0;
  virtual fs::path exportGraphToFile(const json &options) = 0;
  virtual json learnFromDialog(const json &payload, const json &result) = 0;
  virtual std::unordered_map<std::string, double>
  mapWordsToMemes(const std::vector<std::string> &words) = 0;
  virtual std::vector<std::string> getMemeWords(const std::string &memeId) = 0;
  virtual void graphLink(const std::string &fromId, const std::string &toId,
                         double weight, int direction) = 0;
  virtual const Config &config() const = 0;
};
