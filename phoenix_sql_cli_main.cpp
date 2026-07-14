/* phoenix_sql_cli_main.cpp - SQL CLI main entry point */

#include "phoenix_sql_cli.hpp"

#include <iostream>

int main(int argc, char **argv)
{
    return phoenix_sql_cli::runMain(argc, argv, std::cout, std::cerr);
}