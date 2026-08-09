/* Auto-generated from main.cpp
   Segment: struct_Config
   Original lines: 991-1060
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

struct Config {
  fs::path baseDir;
  std::string gatewayHost;
  int portGateway{5080};
  int portStudy{5081};
  int aiCount{7};
  int groupCount{3};
  int groupSize{7};
  int sparkNumAI{7};
  std::string sparkBudget{"default"};
  bool singleProc{false};
  bool groupProc{false};
  int groupProcTimeoutMs{12000};
  bool inferMP{false};
  int inferWorkers{1};
  int redisConnectTimeoutMs{1500};
  std::string redisUrl{"redis://127.0.0.1:6379"};
  std::string redisChannel{"AI-model-workspace"};
  fs::path dbPath;
  int redisCacheDb{1};
  std::string redisCachePrefix{"AI079"};
  fs::path snapshotDir;
  fs::path workerFile;
  int maxWorkers{1};
  fs::path shardCache;
  fs::path lmdbRoot;
  std::size_t lmdbMapSizeBytes{4ull * 1024ull * 1024ull * 1024ull};
  std::string searchEndpoint;
  fs::path robotsDir;
  fs::path lemmaCsv;
  bool lemmaAutoload{false};
  std::size_t lemmaMaxBytes{64ull * 1024ull * 1024ull};
  bool lemmaForce{false};
  int robotsWarmupLimit{200};
  bool robotsAutoload{true};
  bool robotsWarmupShuffle{false};
  int robotsChunkMinWords{2};
  int robotsChunkMaxWords{5};
  int kvmCacheMaxEntries{50000};
  bool learningWarmup{false};
  bool syncStandbyOnBoot{false};
  bool testsAutoload{true};
  fs::path testsDir;
  bool disableBarrier{false};
  bool disableRL{false};
  bool disableADV{false};
  bool disableLearning{false};
  bool disableGnnModule{false};
  bool disableContextModule{false};
  bool usingOllama{true};
  bool ollamaFineTuning{true};
  bool inferenceEnabled{true};
  std::string transformerBackend{"ollama"};
  std::string ollamaModel;
  std::string ollamaBaseUrl{"http://127.0.0.1:11434"};
  int ollamaRequestTimeoutMs{120000};
  std::string llamaCppModel;
  std::string llamaCppBaseUrl{"http://127.0.0.1:8080"};
  int llamaCppRequestTimeoutMs{120000};
  std::string llamaCppLoraFiles;
  bool llamaCppLoraInitWithoutApply{false};
  bool llamaCppFineTuning{false};
  std::string bitnetModel;
  std::string bitnetBaseUrl{"http://127.0.0.1:8090"};
  int bitnetRequestTimeoutMs{120000};
  std::string tinyllamaModel;
  std::string tinyllamaBaseUrl{"http://127.0.0.1:8085"};
  int tinyllamaRequestTimeoutMs{120000};
  bool tinyllamaEnabled{false};
  bool conceptMatrixEnabled{false};
  bool conceptMatrixClearPerRequest{true};
  fs::path ggufModelsDir;
  bool externalBackendStyleSim{false};
  bool transformerBootstrap{false};
  int transformerBootstrapDocs{256};
  std::string logMode{"release"};
  fs::path exportDir;
  bool httpLog{false};
  fs::path tunedConfigPath;
};
