/*----------------------------------------------------------------------------

  Random Number Generator.

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
/*------------------------- Random Number Generator --------------------------*/
/*----------------------------------------------------------------------------*/
/** @file rand.c
    Random number generator functions.
    @author rafael grompone von gioi (grompone@gmail.com)
 */
/*----------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

/*----------------------------------------------------------------------------*/
/** Uniform random number generator in [1,2147483562].

    This algorithm is described and analyzed on:

      "Efficient and Portable Combined Random Number Generators"
      by Pierre L'Ecuyer, Communications of the ACM, vol. 31, num. 6,
      pp. 742-749 and 774, June 1988.

    This code is a modification from the code available at
      http://cg.scs.carleton.ca/~luc/lecuyer.c
    on February 5, 2011.

    The algorithm is based on two combined Multiplicative Linear
    Congruential Generators (MLCGs). The generator produces double
    numbers in the range (0,1), (it will never return 0.0 or 1.0).
    The period of the generator is about 2.30584 x 10^18,
    in the order of 2^61.

    A normal call is in the form 'uniform_rand(NULL)'.
    But it can also be called giving a pointer to long with a seed:
      long seed = 263287632;
      uniform_rand(&seed);
    This is usually done only once. The function 'rand_time_seed'
    performs this initilization with the current time.

    The state of the generator is determined by the static variables
    s1 and s2, that must take values in the following intervals:

      s1 in [1,2147483562]
      s2 in [1,2147483398]
 */
long random_number(long * seed)
{
  static long s1 = 55555;
  static long s2 = 99999;
  long k,z;

  /* Initialization of the seed, if demanded. */
  if( seed != NULL ) s1 = 1 + (*seed % 2147483561); /* s1 in [1,2147483562] */

  /* First MLCG: s1 = (40014 * s1) mod 2147483563, efficient implementation */
  k = s1 / 53668;
  s1 = 40014 * ( s1 % 53668 ) - k * 12211;
  if( s1 < 0 ) s1 += 2147483563;

  /* Second MLCG: s2 = (40692 * s2) mod 2147483399, efficient implementation */
  k = s2 / 52774;
  s2 = 40692 * ( s2 % 52774 ) - k * 3791;
  if( s2 < 0 ) s2 += 2147483399;

  /* Combination of both MLCGs */
  z = ( s1 - 2147483563 ) + s2;
  if(z < 1) z += 2147483562;

  return z;
}

/*----------------------------------------------------------------------------*/
/** Uniform random number generator in (0,1).
 */
double uniform_rand(long * seed)
{
  static double factor = 1.0/2147483563.0;
  return ( (double) random_number(seed) ) * factor;
}

/*----------------------------------------------------------------------------*/
/** Initialize seed with the current time.
 */
void rand_time_seed(void)
{
  long s;
  s = time(NULL);
  random_number(&s);
}

/*----------------------------------------------------------------------------*/
