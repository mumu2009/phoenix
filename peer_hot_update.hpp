#pragma once

#include "peer_evolve_extract.hpp"

#include <string>

namespace phoenix {
namespace peer {

/* Compile the function this process just wrote and LoadLibrary it in the
   peer process. No-op unless AI_PEER_URL is set and the goal says PEER-EVOLVE. */
/* Idle: nothing to teach (no peer, no function, or the peer process
   itself could not be reached). Rejected: their function did not compile
   or did not run. Loaded: the other process accepted the DLL. */
struct PeerPushResult {
  enum class Kind { Idle, Rejected, Loaded };
  Kind kind{Kind::Idle};
  std::string detail;
};

PeerPushResult pushEvolveToPeer(const std::string &goal,
                                const std::string &deliverable);

/* The function the other process last loaded into this one. Empty if
   none has been loaded yet. */
std::string latestIncomingSource();

}  // namespace peer
}  // namespace phoenix
