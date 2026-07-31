/* version.hpp - Phoenix version and codename constants
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

#ifndef PHOENIX_VERSION_HPP
#define PHOENIX_VERSION_HPP

namespace phoenix {

/* Major/minor/patch version numbers for the Phoenix runtime. */
constexpr int PHOENIX_VERSION_MAJOR = 7;
constexpr int PHOENIX_VERSION_MINOR = 0;
constexpr int PHOENIX_VERSION_PATCH = 0;

/* Human-readable codename for this release (v7.0 = Arthur). */
constexpr char PHOENIX_CODENAME[] = "Arthur";

/* Combined version string used by /api/system/status and logs. */
constexpr char PHOENIX_VERSION_STRING[] = "7.0.0 \"Arthur\"";

} // namespace phoenix

#define PHOENIX_VERSION_STRING_LITERAL "7.0.0 \"Arthur\""

#endif // PHOENIX_VERSION_HPP
