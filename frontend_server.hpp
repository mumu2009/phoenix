/* frontend_server.hpp - Frontend static resource server setup
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

#ifndef FRONTEND_SERVER_HPP
#define FRONTEND_SERVER_HPP

#include <drogon/drogon.h>
#include <string>

/* Setup the frontend static resource server on port 5081 */
void setupFrontendServer();

#endif // FRONTEND_SERVER_HPP