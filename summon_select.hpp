/* Deprecated shim.

   EndpointStats + EndpointRanker now live in summon_registry.hpp, right above
   SummonRegistry, because the registry itself uses them (a separate header
   would have formed an include cycle: summon_registry -> summon_select ->
   summon_registry).  Kept as a forwarding header so older includes still
   compile; new code should include "summon_registry.hpp". */
#pragma once

#include "summon_registry.hpp"
