#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

// this example illustrates how to tile a range multiple times
// examples:
//   tiled_range([0, 1, 2, 3], 1) -> [0, 1, 2, 3]
//   tiled_range([0, 1, 2, 3], 2) -> [0, 1, 2, 3, 0, 1, 2, 3]
//   tiled_range([0, 1, 2, 3], 3) -> [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
//   ...

template <typename Iterator>
class tiled_range
{
public:
    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct tile_functor : public thrust::unary_function<difference_type, difference_type>
    {
        difference_type tile_size;

        tile_functor(difference_type tile_size)
            : tile_size(tile_size) {}

        __host__ __device__
            difference_type
            operator()(const difference_type &i) const
        {
            return i % tile_size;
        }
    };

    typedef typename thrust::counting_iterator<difference_type> CountingIterator;
    typedef typename thrust::transform_iterator<tile_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator, TransformIterator> PermutationIterator;

    // type of the tiled_range iterator
    typedef PermutationIterator iterator;

    // construct repeated_range for the range [first,last)
    tiled_range(Iterator first, Iterator last, difference_type tiles)
        : first(first), last(last), tiles(tiles) {}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), tile_functor(last - first)));
    }

    iterator end(void) const
    {
        return begin() + tiles * (last - first);
    }

protected:
    Iterator first;
    Iterator last;
    difference_type tiles;
};