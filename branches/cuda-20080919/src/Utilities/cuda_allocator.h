#ifndef CUDA_ALLOCATOR_H
#define CUDA_ALLOCATOR_H

#ifdef QMC_CUDA
  #include <cuda_runtime_api.h>
#endif
#include <malloc.h>
#include <iostream>

template<typename T> class cuda_allocator;

template<>
class cuda_allocator<void>
{
public:
  typedef void*  pointer;
  typedef const  void* const_pointer;
  typedef void value_type;

  template<class T1> struct rebind 
  { 
    typedef cuda_allocator<T1> other; 
  };
};


template<typename T> 
class cuda_allocator 
{
public:
  typedef size_t    size_type;
  typedef ptrdiff_t difference_type;
  typedef T*        pointer;
  typedef const T*  const_pointer;
  typedef T&        reference;
  typedef const T&  const_reference;
  typedef T         value_type;
  template<typename U> struct rebind { typedef cuda_allocator<U> other; };
  
  cuda_allocator() throw() { }
  cuda_allocator(const cuda_allocator&) throw() { }
  template<typename U> cuda_allocator(const cuda_allocator<U>&) throw() { }
  ~cuda_allocator() throw() { };

  pointer address(reference x) const 
  { return &x; }

  const_pointer address(const_reference x) const 
  { return &x; }
  
  pointer allocate(size_type s, cuda_allocator<void>::const_pointer hint = 0)
  {
#ifdef QMC_CUDA   
    pointer mem;
    cudaMalloc ((void**)&mem, s*sizeof(T));
    return mem;
#else
    return malloc(s*sizeof(T));
#endif
  }
  
  void deallocate(pointer p, size_type n)
  {
    cudaFree (p);
  }

  size_type max_size() const throw()
  { return (size_type)1 << 32 - 1; }
  
  void construct(pointer p, const T& val)
  { 
    //new(static_cast<void*>(p)) T(val);  
  }
  
  void destroy(pointer p) {
    p->~T();
  }
};

#include <vector>

template<typename T>
class cuda_vector : public std::vector<T, cuda_allocator<T> >
{
public:
  cuda_vector(const cuda_vector<T> &vec)
  {
    if (this->size() != vec.size())
      resize(vec.size());
    if (this->size()) 
      cudaMemcpy (&(this[0]), &(vec[0]), this->size()*sizeof(T),
		  cudaMemcpyDeviceToDevice);
  }

  cuda_vector& 
  operator=(const cuda_vector<T> &vec)
  {
    if (this->size() != vec.size())
      resize(vec.size());
    cudaMemcpy (&((*this)[0]), &(vec[0]), this->size()*sizeof(T), 
		cudaMemcpyHostToDevice);
    return *this;
  }

  cuda_vector& 
  operator=(const std::vector<T,std::allocator<T> > &vec)
  {
    if (this->size() != vec.size())
      resize(vec.size());
    cudaMemcpy (&((*this)[0]), &(vec[0]), this->size()*sizeof(T), 
		cudaMemcpyHostToDevice);
    return *this;
  }


};




#endif
