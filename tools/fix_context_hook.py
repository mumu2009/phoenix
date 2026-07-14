import sys

p = r"d:\_phoenix\_079\v6.0Alixander\v6.0Alixander\main_hub_parts\116_section_tail.inc"
with open(p, "rb") as f:
    s = f.read().decode("utf-8")

old = '''            // Hook: Inject concat.history from frontend if contextHint is empty and sessionId exists
            std::cout << "[gateway-hook] Check: hint.text.empty=" << hint.text.empty() 
                      << ", hasSessionId=" << body.contains("sessionId") << std::endl;
            if (!contextModuleDisabled && hint.text.empty() && body.contains("sessionId") && body["sessionId"].is_string()) {
              std::string sessionId = body["sessionId"].get<std::string>();
              std::string contextMode = body.value("contextMode", "auto");
              std::cout << "[gateway-hook] Triggered: sessionId=" << sessionId << ", contextMode=" << contextMode << std::endl;
              // Always try to inject concat.history regardless of mode
              try {
                // Request frontend context status to get concat.history
                auto client = drogon::HttpClient::newHttpClient("http://127.0.0.1:5081");
                if (client) {
                  auto req = drogon::HttpRequest::newHttpRequest();
                  req->setMethod(drogon::Get);
                  req->setPath("/context/status?sessionId=" + sessionId);
                  req->addHeader("Authorization", "Bearer local-dev");

                  auto sharedPromise = std::make_shared<
                      std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>>>();
                  auto future = sharedPromise->get_future();
                  client->sendRequest(req, [sharedPromise](drogon::ReqResult result,
                                                            const drogon::HttpResponsePtr &resp) {
                    try {
                      sharedPromise->set_value({result, resp});
                    } catch (...) {
                    }
                  });

                  if (future.wait_for(std::chrono::milliseconds(1000)) == std::future_status::ready) {
                    auto result = future.get();
                    if (result.first == drogon::ReqResult::Ok && result.second) {
                      auto bodyStr = result.second->getBody();
                      if (!bodyStr.empty()) {
                        json contextJson = json::parse(bodyStr);
                        if (contextJson.contains("concat") && contextJson["concat"].contains("history") &&
                            contextJson["concat"]["history"].is_array() && !contextJson["concat"]["history"].empty()) {
                          std::ostringstream concatHint;
                          concatHint << "【对话历史】\\n";
                          for (const auto &hist : contextJson["concat"]["history"]) {
                            if (hist.contains("text") && hist["text"].is_string()) {
                              std::string role = "User";
                              if (hist.contains("role") && hist["role"].is_string())
                                role = hist["role"].get<std::string>();
                              concatHint << role << ": " << hist["text"].get<std::string>() << "\\n";
                            }
                          }
                          hint.text = concatHint.str();
                          hint.mode = "short";
                          hint.weight = 0.95;
                          std::cout << "[gateway-hook] Injected concat.history for sessionId=" << sessionId
                                    << ", historySize=" << contextJson["concat"]["history"].size() << std::endl;
                        }
                      }
                    }
                  }
                }
              } catch (const std::exception &e) {
                std::cout << "[gateway-hook] Failed to fetch concat.history: " << e.what() << std::endl;
              }
            }
            
            // Keep user-provided contextHint even when the context module is disabled;
            // this allows callers to inject their own history without enabling the module.
            auto gnnHint = buildGnnHintSummary(graphResult, body, hint);
            hint = harmonizeHintWithGnn(hint, gnnHint.align);
            if (gnnModuleDisabled)
              gnnHint = GnnHintSummary{};
            std::string sparkAnn;
            (void)sparkAnn;
            if (!sparkAnn.empty()) {
              if (graphContext.empty())
                graphContext = sparkAnn;
              else
                graphContext = graphContext + "\\n" + sparkAnn;
            }
            if (!contextModuleDisabled) {
              text = applyContextHintToText(text, hint);
              graphContext = applyContextHintToGraph(graphContext, hint);
            } else {
              graphContext.clear();
              // User-provided contextHint is still applied when context module is disabled
              if (!hint.text.empty()) {
                text = applyContextHintToText(text, hint);
              }
            }'''

new = '''            // Hook: fetch compressed context hint from frontend if not explicitly provided.
            std::cout << "[gateway-hook] Check: hint.text.empty=" << hint.text.empty()
                      << ", hasSessionId=" << body.contains("sessionId") << std::endl;
            if (hint.text.empty() && body.contains("sessionId") && body["sessionId"].is_string()) {
              std::string sessionId = body["sessionId"].get<std::string>();
              std::string contextMode = body.value("contextMode", "auto");
              std::cout << "[gateway-hook] Triggered: sessionId=" << sessionId << ", contextMode=" << contextMode << std::endl;
              try {
                auto client = drogon::HttpClient::newHttpClient("http://127.0.0.1:5081");
                if (client) {
                  json requestJson;
                  requestJson["sessionId"] = sessionId;
                  requestJson["text"] = text;
                  requestJson["mode"] = contextMode;
                  auto req = drogon::HttpRequest::newHttpRequest();
                  req->setMethod(drogon::Post);
                  req->setPath("/context/hint");
                  req->addHeader("Authorization", "Bearer local-dev");
                  req->setContentTypeCode(drogon::CT_APPLICATION_JSON);
                  req->setBody(requestJson.dump());

                  auto sharedPromise = std::make_shared<
                      std::promise<std::pair<drogon::ReqResult, drogon::HttpResponsePtr>>>();
                  auto future = sharedPromise->get_future();
                  client->sendRequest(req, [sharedPromise](drogon::ReqResult result,
                                                            const drogon::HttpResponsePtr &resp) {
                    try {
                      sharedPromise->set_value({result, resp});
                    } catch (...) {
                    }
                  });

                  if (future.wait_for(std::chrono::milliseconds(1000)) == std::future_status::ready) {
                    auto result = future.get();
                    if (result.first == drogon::ReqResult::Ok && result.second) {
                      auto bodyStr = result.second->getBody();
                      if (!bodyStr.empty()) {
                        json contextJson = json::parse(bodyStr, nullptr, false);
                        if (contextJson.is_object() && contextJson.value("ok", false) &&
                            contextJson.contains("contextHint") && contextJson["contextHint"].is_string()) {
                          std::string hintText = contextJson["contextHint"].get<std::string>();
                          if (!hintText.empty()) {
                            hint.text = hintText;
                            hint.mode = contextJson.value("mode", "auto");
                            hint.weight = 0.9;
                            if (contextJson.contains("weight") && contextJson["weight"].is_number())
                              hint.weight = contextJson["weight"].get<double>();
                            std::cout << "[gateway-hook] Injected contextHint for sessionId=" << sessionId
                                      << ", mode=" << hint.mode << ", weight=" << hint.weight << std::endl;
                          }
                        }
                      }
                    }
                  }
                }
              } catch (const std::exception &e) {
                std::cout << "[gateway-hook] Failed to fetch contextHint: " << e.what() << std::endl;
              }
            }

            auto gnnHint = buildGnnHintSummary(graphResult, body, hint);
            hint = harmonizeHintWithGnn(hint, gnnHint.align);
            if (gnnModuleDisabled)
              gnnHint = GnnHintSummary{};
            std::string sparkAnn;
            (void)sparkAnn;
            if (!sparkAnn.empty()) {
              if (graphContext.empty())
                graphContext = sparkAnn;
              else
                graphContext = graphContext + "\\n" + sparkAnn;
            }
            // Always apply session-level context hint; only mix graph context when context module is enabled.
            if (!contextModuleDisabled) {
              text = applyContextHintToText(text, hint);
              graphContext = applyContextHintToGraph(graphContext, hint);
            } else {
              graphContext.clear();
              text = applyContextHintToText(text, hint);
            }'''

if old not in s:
    print("OLD hook block not found; nothing changed.")
    sys.exit(0)

s = s.replace(old, new)
with open(p, "wb") as f:
    f.write(s.encode("utf-8"))
print("Replaced gateway context hook in", p)
