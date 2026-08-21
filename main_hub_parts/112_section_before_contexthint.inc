// Auto-generated from main.cpp
// Segment: section_before_ContextHint
// Original lines: 13884-14002

    if (config_.httpLog)
      std::cout << "[chat] " << provider << " start baseUrl=" << baseUrl
                << " model=" << model << " timeoutMs=" << timeoutMs << std::endl;
    out["provider"] = provider;
    try {
      std::string selectedModel = trimCopy(model);
      if (selectedModel.empty())
        selectedModel = provider;
      out["model"] = selectedModel;

      // If the gateway already prepended the context hint to the user text
      // (via applyContextHintToText), split it out so llama_proxy can inject it
      // as a system prompt rather than duplicating it inside the user message.
      std::string contextHintText;
      std::string prompt = text;
      if (!text.empty()) {
        const std::string prefix = "[Context hint ";
        if (text.compare(0, prefix.size(), prefix) == 0) {
          auto wrapperEnd = text.find(']', prefix.size());
          if (wrapperEnd != std::string::npos) {
            auto summaryEnd = text.find("]\n", wrapperEnd);
            if (summaryEnd != std::string::npos) {
              contextHintText =
                  text.substr(wrapperEnd + 1, summaryEnd - wrapperEnd);
              prompt = text.substr(summaryEnd + 2);
            }
          }
        }
      }

      size_t maxContextChars = static_cast<size_t>(
          phoenix::cfgOr<int>("context.maxContextChars", 65536));
      std::string cappedGraphContext = graphContext;
      if (cappedGraphContext.size() > maxContextChars) {
        size_t start = cappedGraphContext.size() - (maxContextChars - 25);
        auto nl = cappedGraphContext.find('\n', start);
        if (nl != std::string::npos)
          start = nl + 1;
        // Make sure we start at a valid UTF-8 code-point boundary, otherwise
        // the resulting prompt can contain invalid multi-byte sequences and
        // nlohmann::json::dump() will throw a type_error.316.
        while (start < cappedGraphContext.size() &&
               (static_cast<unsigned char>(cappedGraphContext[start]) & 0xC0) == 0x80) {
          ++start;
        }
        cappedGraphContext =
            std::string("... [context truncated]\n") + cappedGraphContext.substr(start);
      }

      if (contextHintText.empty() && !cappedGraphContext.empty())
        prompt =
            std::string("Context:\n") + cappedGraphContext + "\n\nUser:\n" + text;

      // Ensure the prompt is valid UTF-8 so nlohmann::json can dump it without
      // throwing a type_error.316 in the middle of the request.
      auto sanitizeUtf8 = [](const std::string &s) -> std::string {
        std::string out;
        out.reserve(s.size());
        size_t i = 0;
        while (i < s.size()) {
          unsigned char c = static_cast<unsigned char>(s[i]);
          size_t len = 1;
          if (c >= 0xC0 && c <= 0xDF) len = 2;
          else if (c >= 0xE0 && c <= 0xEF) len = 3;
          else if (c >= 0xF0 && c <= 0xF7) len = 4;
          else if (c >= 0x80) { out.push_back('?'); ++i; continue; }
          if (i + len > s.size()) { out.push_back('?'); ++i; continue; }
          bool valid = true;
          for (size_t j = 1; j < len; ++j) {
            unsigned char cb = static_cast<unsigned char>(s[i + j]);
            if ((cb & 0xC0) != 0x80) { valid = false; break; }
          }
          if (valid) {
            out.append(s, i, len);
            i += len;
          } else {
            out.push_back('?');
            ++i;
          }
        }
        return out;
      };
      prompt = sanitizeUtf8(prompt);

      json options = json{{"num_predict", std::max(1, maxTokens)}};
      if (inferenceOptions.is_object() && !inferenceOptions.empty()) {
        for (auto it = inferenceOptions.begin(); it != inferenceOptions.end(); ++it) {
          if (it.value().is_number())
            options[it.key()] = it.value();
        }
      }
      json payload = {
          {"model", selectedModel},
          {"stream", false},
          {"messages",
           json::array({json{{"role", "user"}, {"content", prompt}}})},
          {"options", options}};
      if (!contextHintText.empty())
        payload["contextHint"] = sanitizeUtf8(contextHintText);

      auto endpoint = trimCopy(baseUrl);
      if (endpoint.empty())
        endpoint = "http://127.0.0.1:8080";

      // Parse endpoint: http://host:port
      std::string host = "127.0.0.1";
      int port = 8080;
      {
        std::string ep = endpoint;
        if (ep.substr(0, 7) == "http://") ep = ep.substr(7);
        auto col = ep.find(':');
        if (col != std::string::npos) {
          host = ep.substr(0, col);
          port = std::stoi(ep.substr(col + 1));
        } else {
          host = ep;
          port = 80;
        }
      }

      std::string bodyStr = payload.dump();
      std::ostringstream httpReq;
      httpReq << "POST /api/chat HTTP/1.1\r\n"
              << "Host: " << host << ":" << port << "\r\n"
              << "Content-Type: application/json\r\n"
              << "Content-Length: " << bodyStr.size() << "\r\n"
              << "Connection: close\r\n"
              << "\r\n"
              << bodyStr;
      std::string reqStr = httpReq.str();

      // Synchronous TCP send/recv using Winsock with full connect() timeout
      std::string rawResponse;
      {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2,2), &wsa);
        SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock == INVALID_SOCKET) {
          out["ok"] = false; out["error"] = provider + " socket create failed"; out["reply"] = "";
          return out;
        }
        int timeoutVal = std::max(1000, timeoutMs);

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<unsigned short>(port));
        inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

        // Non-blocking connect with select() timeout so connect() never hangs forever
        ioctl_arg_t nonblk = 1;
        ioctlsocket(sock, FIONBIO, &nonblk);
        int cr = connect(sock, (struct sockaddr*)&addr, sizeof(addr));
        bool connected = false;
        if (cr == SOCKET_ERROR &&
            (WSAGetLastError() == WSAEWOULDBLOCK || WSAGetLastError() == EINPROGRESS)) {
          fd_set wfds;
          FD_ZERO(&wfds);
          FD_SET(sock, &wfds);
          struct timeval tv;
          tv.tv_sec  = timeoutVal / 1000;
          tv.tv_usec = (timeoutVal % 1000) * 1000;
          if (select((int)sock + 1, nullptr, &wfds, nullptr, &tv) == 1) {
            int err = 0; sockopt_len_t errlen = sizeof(err);
            getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&err, &errlen);
            connected = (err == 0);
          }
        } else if (cr == 0) {
          connected = true;
        }
        if (!connected) {
          closesocket(sock);
          out["ok"] = false;
          int lastErr = WSAGetLastError();
          out["error"] = provider + " connect failed to " + host + ":" + std::to_string(port) +
                         " (err=" + std::to_string(lastErr) + ")";
          out["reply"] = "";
          if (config_.httpLog)
            std::cout << "[chat] " << provider << " connect failed " << host << ":" << port
                      << " err=" << lastErr << std::endl;
          return out;
        }
        // Restore blocking mode and set send/recv timeouts
        ioctl_arg_t blk = 0;
        ioctlsocket(sock, FIONBIO, &blk);
#ifndef _WIN32
        {
          struct timeval tv;
          tv.tv_sec = timeoutVal / 1000;
          tv.tv_usec = (timeoutVal % 1000) * 1000;
          setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
          setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        }
#else
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeoutVal, sizeof(timeoutVal));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeoutVal, sizeof(timeoutVal));
#endif

        int sent = 0;
        while (sent < (int)reqStr.size()) {
          int n = send(sock, reqStr.c_str() + sent, (int)reqStr.size() - sent, 0);
          if (n <= 0) break;
          sent += n;
        }
        char buf[4096];
        while (true) {
          int n = recv(sock, buf, sizeof(buf), 0);
          if (n <= 0) break;
          rawResponse.append(buf, n);
        }
        closesocket(sock);
      }

      // Split HTTP response: headers + body
      auto hdrEnd = rawResponse.find("\r\n\r\n");
      std::string httpBody;
      int httpStatus = 0;
      if (hdrEnd != std::string::npos) {
        httpBody = rawResponse.substr(hdrEnd + 4);
        // Parse status line: HTTP/1.1 200 OK
        auto sp1 = rawResponse.find(' ');
        if (sp1 != std::string::npos) {
          auto sp2 = rawResponse.find(' ', sp1 + 1);
          try { httpStatus = std::stoi(rawResponse.substr(sp1 + 1, sp2 - sp1 - 1)); } catch(...) {}
        }
        // Handle chunked transfer encoding
        if (rawResponse.find("Transfer-Encoding: chunked") != std::string::npos ||
            rawResponse.find("transfer-encoding: chunked") != std::string::npos) {
          std::string decoded;
          size_t pos = 0;
          while (pos < httpBody.size()) {
            auto crlf = httpBody.find("\r\n", pos);
            if (crlf == std::string::npos) break;
            size_t chunkSize = 0;
            try { chunkSize = std::stoul(httpBody.substr(pos, crlf - pos), nullptr, 16); } catch(...) { break; }
            if (chunkSize == 0) break;
            pos = crlf + 2;
            if (pos + chunkSize > httpBody.size()) break;
            decoded.append(httpBody, pos, chunkSize);
            pos += chunkSize + 2;
          }
          httpBody = decoded;
        }
      } else if (!rawResponse.empty()) {
        httpBody = rawResponse;
        httpStatus = 200;
      }

      if (httpStatus < 200 || httpStatus >= 300) {
        out["ok"] = false;
        out["status"] = httpStatus;
        out["error"] = provider + " bad status " + std::to_string(httpStatus);
        out["reply"] = "";
        if (config_.httpLog)
          std::cout << "[chat] " << provider << " bad status=" << httpStatus
                    << " body=" << httpBody.substr(0, 200) << std::endl;
        return out;
      }

      json doc = json::parse(httpBody, nullptr, false);
      if (doc.is_discarded()) {
        out["ok"] = false;
        out["error"] = provider + " invalid json";
        out["reply"] = "";
        return out;
      }

      std::string reply;
      if (doc.contains("message") && doc["message"].is_object() &&
          doc["message"].contains("content") &&
          doc["message"]["content"].is_string())
        reply = doc["message"]["content"].get<std::string>();
      else if (doc.contains("response") && doc["response"].is_string())
        reply = doc["response"].get<std::string>();

      out["ok"] = !reply.empty();
      out["reply"] = reply;
      if (config_.httpLog)
        std::cout << "[chat] " << provider << " replyLen=" << reply.size() << " ok=" << out["ok"]
                  << std::endl;
      return out;
    } catch (const std::exception &e) {
      out["ok"] = false;
      out["error"] = std::string("exception: ") + e.what();
      out["reply"] = "";
      if (config_.httpLog)
        std::cout << "[chat] " << provider << " exception=" << e.what() << std::endl;
      return out;
    } catch (...) {
      out["ok"] = false;
      out["error"] = provider + " unknown error";
      out["reply"] = "";
      if (config_.httpLog)
        std::cout << "[chat] " << provider << " unknown exception" << std::endl;
      return out;
    }
  }

  // Whether chatWithLlamaCpp() should drive the patched llama-server's
  // split backend, now through the server-side /phx/generate endpoint
  // (phoenix::v7::llamaSplitChat), instead of its native, unmodified
  // /apply-template + /v1/chat/completions text endpoints
  // (phoenix::v7::llamaTextOnlyChat). Controlled by
  // main.inference.llamaUseSplitBackend in config/phoenix.json (defaults to
  // false). The llama_server_mods/enc_dec_separation.patch has been updated
  // to make the split endpoints stable for LLM_ARCH_LLAMA models; the
  // llamaTextOnlyChat path remains the fallback when the split backend is
  // disabled or fails.
  bool useLlamaSplitBackend() const {
    if (config_.llamaCppBaseUrl.empty()) return false;
    bool enabled = false;
    try {
      enabled = phoenix::cfgOr<bool>("main.inference.llamaUseSplitBackend", false);
    } catch (...) {}
    return enabled;
  }

  json callLlamaCppBackend(const std::string &text,
                           const std::string &graphContext, int maxTokens,
                           const json &inferenceOptions) {
    json r = useLlamaSplitBackend()
                 ? phoenix::v7::llamaSplitChat(
                       config_.llamaCppBaseUrl, config_.llamaCppRequestTimeoutMs,
                       config_.llamaCppModel, text, graphContext, maxTokens,
                       inferenceOptions)
                 : phoenix::v7::llamaTextOnlyChat(
                       config_.llamaCppBaseUrl, config_.llamaCppRequestTimeoutMs,
                       config_.llamaCppModel, text, graphContext, maxTokens,
                       inferenceOptions);
    if (config_.httpLog)
      std::cout << "[chat] llamacpp(" << (useLlamaSplitBackend() ? "split" : "text")
                << ") ok=" << r.value("ok", false)
                << " replyLen=" << r.value("reply", std::string()).size()
                << std::endl;
    return r;
  }

  json chatWithLlamaCpp(const std::string &text,
                        const std::string &graphContext, int maxTokens,
                        const json &inferenceOptions = json::object()) {
    std::lock_guard<std::mutex> llmLock(llamaCallMu_);
    bool voterEnabled = false;
    int numPasses = 5;
    double maskRate = 0.05;
    try {
      voterEnabled = phoenix::cfgOr<bool>("llama_voter.enabled", false);
    } catch (...) {}
    if (voterEnabled) {
      try { numPasses = phoenix::cfgOr<int>("llama_voter.numPasses", 5); } catch (...) {}
      try { maskRate = phoenix::cfgOr<double>("llama_voter.maskRate", 0.05); } catch (...) {}
    }
    if (!voterEnabled || numPasses <= 1 || maskRate <= 0.0) {
      return callLlamaCppBackend(text, graphContext, maxTokens, inferenceOptions);
    }

    auto maskTokens = [&](const std::string &s,
                          std::mt19937 &rng) -> std::string {
      auto toks = tokenize(s);
      if (toks.empty()) return s;
      std::vector<std::string> out;
      out.reserve(toks.size());
      std::uniform_real_distribution<double> dist(0.0, 1.0);
      for (auto &t : toks) {
        if (dist(rng) >= maskRate) out.push_back(t);
      }
      if (out.empty()) out.push_back(toks[rng() % toks.size()]);
      std::string joined;
      for (size_t i = 0; i < out.size(); ++i) {
        if (i) joined += ' ';
        joined += out[i];
      }
      return joined;
    };

    std::vector<json> candidates;
    candidates.reserve(numPasses);
    std::random_device rd;
    std::mt19937 rng(rd());
    auto reqEmb = textToMiniEmbedding(text, 64);
    json best;
    double bestScore = -1e9;
    for (int i = 0; i < numPasses; ++i) {
      std::string mtext = (i == 0) ? text : maskTokens(text, rng);
      std::string mctx = (i == 0) ? graphContext : maskTokens(graphContext, rng);
      json r = callLlamaCppBackend(mtext, mctx, maxTokens, inferenceOptions);
      if (!r.is_object()) continue;
      std::string reply = r.value("reply", "");
      if (reply.empty()) continue;
      auto repEmb = textToMiniEmbedding(reply, 64);
      double affinity = cosineSim(reqEmb, repEmb);
      double lenNorm = 1.0 / (1.0 + std::abs((double)reply.size() - (double)text.size()) / 100.0);
      double score = 0.7 * clamp01((affinity + 1.0) * 0.5) + 0.3 * lenNorm;
      if (score > bestScore) {
        bestScore = score;
        best = r;
      }
      candidates.push_back(std::move(r));
    }

    if (!best.is_object()) {
      return callLlamaCppBackend(text, graphContext, maxTokens, inferenceOptions);
    }
    best["rfVotes"] = static_cast<int>(candidates.size());
    best["rfBestScore"] = bestScore;
    best["rfMethod"] = "llama-server-random-forest-micro-mask";
    return best;
  }

  json chatWithBitNet(const std::string &text, const std::string &graphContext,
                      int maxTokens,
                      const json &inferenceOptions = json::object()) {
    return chatWithExternalAdapter(
        "bitnet", config_.bitnetBaseUrl, config_.bitnetModel,
        config_.bitnetRequestTimeoutMs, text, graphContext, maxTokens,
        inferenceOptions);
  }

