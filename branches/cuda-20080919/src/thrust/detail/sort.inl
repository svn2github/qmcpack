/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file sort.inl
 *  \brief Inline file for sort.h.
 */

#include <thrust/sort.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/functional.h>
#include <thrust/detail/dispatch/sort.h>

namespace thrust
{

///////////////
// Key Sorts //
///////////////

template<typename RandomAccessIterator>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;

    // default comparison method is less<KeyType>
    thrust::sort(first, last, thrust::less<KeyType>());
}

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void sort(RandomAccessIterator first,
            RandomAccessIterator last,
            StrictWeakOrdering comp)
{
    // dispatch on space
    thrust::detail::dispatch::sort(first, last, comp,
            typename thrust::iterator_space<RandomAccessIterator>::type());
}

template<typename RandomAccessIterator>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator>::value_type KeyType;

    // default comparison method is less<KeyType>
    thrust::stable_sort(first, last, thrust::less<KeyType>());
} 

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_sort(RandomAccessIterator first,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
    // dispatch on space
    thrust::detail::dispatch::stable_sort(first, last, comp,
            typename thrust::iterator_space<RandomAccessIterator>::type());
}



/////////////////////
// Key-Value Sorts //
/////////////////////

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void sort_by_key(RandomAccessKeyIterator keys_first,
                   RandomAccessKeyIterator keys_last,
                   RandomAccessValueIterator values_first)
{
    typedef typename thrust::iterator_traits<RandomAccessKeyIterator>::value_type KeyType;

    // default comparison method is less<KeyType>
    sort_by_key(keys_first, keys_last, values_first, thrust::less<KeyType>());
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void sort_by_key(RandomAccessKeyIterator keys_first,
                   RandomAccessKeyIterator keys_last,
                   RandomAccessValueIterator values_first,
                   StrictWeakOrdering comp)
{
    // XXX forward sort_by_key to stable_sort_by_key
    stable_sort_by_key(keys_first, keys_last, values_first, comp);
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator>
  void stable_sort_by_key(RandomAccessKeyIterator keys_first,
                          RandomAccessKeyIterator keys_last,
                          RandomAccessValueIterator values_first)
{
    typedef typename thrust::iterator_traits<RandomAccessKeyIterator>::value_type KeyType;

    // default comparison method is less<KeyType>
    thrust::stable_sort_by_key(keys_first, keys_last, values_first, thrust::less<KeyType>());
}

template<typename RandomAccessKeyIterator,
         typename RandomAccessValueIterator,
         typename StrictWeakOrdering>
  void stable_sort_by_key(RandomAccessKeyIterator keys_first,
                          RandomAccessKeyIterator keys_last,
                          RandomAccessValueIterator values_first,
                          StrictWeakOrdering comp)
{
    // dispatch on space
    thrust::detail::dispatch::stable_sort_by_key(keys_first, keys_last, values_first, comp,
            typename thrust::iterator_space<RandomAccessKeyIterator>::type(),
            typename thrust::iterator_space<RandomAccessValueIterator>::type());
}

} // last namespace thrust

