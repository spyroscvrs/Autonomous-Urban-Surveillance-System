/*----------------------------------------------------------------------------

  Command Line interface handling functions.

  Copyright 2010-2011 rafael grompone von gioi (grompone@gmail.com)

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

  ----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*--------------------- Command Line interface handling ----------------------*/
/*----------------------------------------------------------------------------*/
/** @file cmd_line_handler.h
    @brief Command line interface.
    @author rafael grompone von gioi (grompone@gmail.com)
 */
/*----------------------------------------------------------------------------*/
#ifndef CMD_LINE_HEADER
#define CMD_LINE_HEADER

/*----------------------------------------------------------------------------*/
#define FIELD_LENGTH 160
#define HELP_OPTION    "--help"
#define VERSION_OPTION "--version"

/*----------------------------------------------------------------------------*/
/** Structure to store one argument definition and read value.
 */
struct argument
{
  char name[FIELD_LENGTH];     /* name to internally identify the argument */
  char desc[FIELD_LENGTH];     /* description */
  char id;                     /* letter used with '-' to use the option */
  char type;                   /* i=int, d=double, s=str, b=bool */
  int required;
  int assigned;
  int def_value;               /* true or false, a default value is assigned? */
  char d_value[FIELD_LENGTH];  /* default value */
  char s_value[FIELD_LENGTH];  /* string found, also the value if 'str' */
  int  i_value;
  double f_value;
  int    min_set;              /* true or false, is minimal value set? */
  double min;
  int    max_set;              /* true or false, is maximal value set? */
  double max;
};

/*----------------------------------------------------------------------------*/
/** Structure to store the full set of argument definitions and its values.
 */
struct arguments
{
  char name[FIELD_LENGTH];
  char author[FIELD_LENGTH];
  char version[FIELD_LENGTH];
  char desc[FIELD_LENGTH];
  char compiled[FIELD_LENGTH];
  char year[FIELD_LENGTH];
  int  arg_num;
  int  arg_allocated;
  struct argument * args;
};

/*----------------------------------------------------------------------------*/
void free_arguments(struct arguments * arg);
void print_version(struct arguments * arg, FILE * f);
void use(struct arguments * arg);
struct arguments * process_arguments(char * desc, int argc, char ** argv);
int is_assigned(struct arguments * arg, char * name);
char * get_str(struct arguments * arg, char * name);
int get_int(struct arguments * arg, char * name);
int * get_int_ptr(struct arguments * arg, char * name);
double get_double(struct arguments * arg, char * name);
double * get_double_ptr(struct arguments * arg, char * name);

#endif /* !CMD_LINE_HEADER */
/*----------------------------------------------------------------------------*/
