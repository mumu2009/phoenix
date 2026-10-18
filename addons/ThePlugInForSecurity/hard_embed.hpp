/* hard_embed.hpp - Process-level defensive controls for the Windows host.

   Soft embedding (meme statistics, identification, inspectText) stays as it
   is. This layer adds controls the operating system actually enforces:
   startup mitigation audit, workspace confinement, and a fail-closed pin
   check before a config or plugin file is trusted. E-stop and MemeBarrier
   are wired in by the host; until that wiring is present, sensitive
   operations refuse. */

#pragma once

#include <cstddef>
#include <string>

namespace phoenix {
namespace secamp {

struct ProcessMitigationReport {
  bool depImage{false};
  bool aslrImage{false};
  bool highEntropyImage{false};
  bool cfgImage{false};
  bool depPolicy{false};
  bool extensionPointsDisabled{false};
  bool noRemoteImages{false};
  bool noLowMandatoryLabelImages{false};
  bool preferSystem32Images{false};
  bool heapTerminateOnCorruption{false};
  bool sealed{false};
  std::string detail;
};

struct PathDecision {
  bool allowed{false};
  std::string reason;
  std::string canonical;
};

struct IntegrityDecision {
  bool allowed{false};
  std::string reason;
  std::string sha256;
};

struct PeFileAudit {
  bool ok{false};
  bool dep{false};
  bool aslr{false};
  bool highEntropy{false};
  bool cfg{false};
  std::string detail;
};

/* Apply safe process mitigations once and audit this image. Idempotent.
   Never throws. sealed is false when a required control is missing. */
const ProcessMitigationReport &ensureProcessHardEmbed();
ProcessMitigationReport hardEmbedReport();
bool mitigationsSealed();

/* MemeBarrier binds these at construction. Null leaves the probe unwired. */
using HardEmbedProbeFn = bool (*)();
void bindEstopProbes(HardEmbedProbeFn latched, HardEmbedProbeFn handlerInstalled);
void notifyMemeBarrierAttached();
bool memeBarrierAttached();
bool estopProbeBound();
bool estopLatched();
bool estopHandlerInstalled();
/* True only when the latch probe is bound and the latch is set. Unwired
   is not a latch: the soft meme barrier must keep its existing decisions. */
bool estopRequiresBlock();

/* True when mitigations are sealed, the barrier has attached, the e-stop
   probe is bound, and the latch is not set. */
bool safetyWiringClosed();

void noteWorkspaceRoot(const std::string &root);
PathDecision workspaceRootAudit();
PathDecision auditWorkspaceRoot(const std::string &root);

/* Write gate. Refuses when mitigations are not sealed, safety wiring is
   incomplete, the e-stop is latched, the root is sensitive, or the
   relative path escapes the root (including via a reparse point). */
PathDecision confineToWorkspace(const std::string &workspaceRoot,
                                const std::string &relative);

bool isSensitivePath(const std::string &path);

/* Hash gate. Empty pin, mismatch, missing file, escape, and sensitive
   locations all refuse. Does not require the process seal so the hash
   itself can be tested; allowArtifactLoad adds the seal and wiring. */
IntegrityDecision verifyPinnedFile(const std::string &path,
                                   const std::string &expectedHex);

/* Fail closed: seal + path hygiene + e-stop wiring + pin. */
IntegrityDecision allowArtifactLoad(const std::string &path,
                                    const std::string &expectedHex);

std::string sha256Bytes(const void *data, std::size_t len);
std::string sha256File(const std::string &path, std::string *error);

PeFileAudit auditPeFile(const std::string &path);

/* Test-only. Does not undo OS mitigations. */
void testOnlyForceUnsealed(bool on);
void testOnlyResetWiring();
void testOnlySetHashCap(std::size_t bytes);

} // namespace secamp
} // namespace phoenix
