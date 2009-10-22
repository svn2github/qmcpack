#ifndef GPU_VECTOR_H
#define GPU_VECTOR_H

#include <malloc.h>
#include <iostream>
#include <string>
#include <map>
#include <vector>

#ifdef QMC_CUDA
#include <cuda_runtime_api.h>
#endif

namespace gpu 
{
  struct gpu_mem_object
  {
    size_t num_objects;
    size_t total_bytes;
    gpu_mem_object (size_t size) :
      num_objects(1), total_bytes(size)
    {
    }
    gpu_mem_object() : num_objects(0), total_bytes(0)
    { }
  };
  
  
  class cuda_memory_manager_type
  {
  private:
    std::map<std::string,gpu_mem_object> gpu_mem_map;
    std::map<void*,std::pair<std::string,size_t> > gpu_pointer_map;

  public:
    void *allocate (size_t bytes, std::string name="");
    
    void deallocate (void *p);

    void report();
  };

  extern cuda_memory_manager_type cuda_memory_manager;

  template<typename T> class host_vector;

  template<typename T>
  class device_vector
  {
  private:
    T *data_pointer;
    size_t current_size, alloc_size;
    std::string name;
  public:
    typedef T* pointer;

    inline
    device_vector() : data_pointer(NULL), current_size(0), alloc_size(0)
    { }

    inline
    device_vector(std::string myName) : name(myName), data_pointer(NULL), 
				   current_size(0), alloc_size(0)
    {  }

    inline
    device_vector(size_t size) : data_pointer(NULL), current_size(0), alloc_size(0)
    {
      resize (size);
    }

    inline 
    device_vector(std::string myName, size_t size) : name(myName), data_pointer(NULL), 
						current_size(0), alloc_size(0)
    {
      resize(size);
    }
  
    inline
    device_vector(const host_vector<T> &vec);
  
    inline T& operator[](size_t i) const
    { return data_pointer[i]; }

    inline void
    resize(size_t size)
    {
      size_t byte_size = sizeof(T)*size;
      if (alloc_size == 0) {
	data_pointer = (T*)cuda_memory_manager.allocate(byte_size, name);
	current_size = alloc_size = size;
      }
      else if (size > alloc_size) {
	cuda_memory_manager.deallocate (data_pointer);
	data_pointer = (T*)cuda_memory_manager.allocate(byte_size, name);
	current_size = alloc_size = size;
      }
      else
	current_size = size;
    }
    
    inline void 
    clear() 
    {
      if (alloc_size) {
	cuda_memory_manager.deallocate (data_pointer);
	data_pointer = NULL;
	current_size = alloc_size = 0;
      }
    }

    inline size_t 
    size() const { return current_size; }
    
    
    inline device_vector& 
    operator=(const device_vector<T> &vec)
    {
      if (this->size() != vec.size())
	resize(vec.size());
#ifdef QMC_CUDA
      cudaMemcpy (data_pointer, &(vec[0]), this->size()*sizeof(T), 
		  cudaMemcpyDeviceToDevice);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
	fprintf (stderr, 
		 "CUDA error in device_vector::operator=(device_vector):\n  %s\n",
		 cudaGetErrorString(err));
	fprintf (stderr, "vec.size() = %ld\n", vec.size());
	abort();
      }
#endif
      return *this;
    }

    device_vector(const device_vector<T> &vec) :
      data_pointer(NULL), current_size(0), alloc_size(0)
    {
      resize(vec.size());
#ifdef QMC_CUDA
      if (this->size() != 0) {
	cudaMemcpy (data_pointer, &(vec[0]), vec.size()*sizeof(T),
		    cudaMemcpyDeviceToDevice);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
	  fprintf (stderr, "CUDA error in device_vector::copy constructor:\n  %s\n",
		   cudaGetErrorString(err));
	  abort();
	}
      }
#endif
    }

    device_vector& 
    operator=(const std::vector<T,std::allocator<T> > &vec)
    {
      if (this->size() != vec.size())
	resize(vec.size());
#ifdef QMC_CUDA
      cudaMemcpy (data_pointer, &(vec[0]), this->size()*sizeof(T), 
		  cudaMemcpyHostToDevice);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
	fprintf (stderr, "CUDA error in device_vector::operator=(vector):\n  %s\n",
		 cudaGetErrorString(err));
	abort();
      }
#endif
      return *this;
    }

    device_vector& 
    operator=(const host_vector<T> &vec)
    {
      if (this->size() != vec.size()) 
	this->resize(vec.size());
#ifdef QMC_CUDA
      cudaMemcpy (&((*this)[0]), &(vec[0]), vec.size()*sizeof(T), 
		  cudaMemcpyHostToDevice);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
	fprintf (stderr, "CUDA error in device_vector::operator=(host_vector):\n  %s\n",
		 cudaGetErrorString(err));
	abort();
      }
#endif
      return *this;
    }
    
    inline T* 
    data() const 
    { return data_pointer; }
    
  };


  
  template<typename T>
  class host_vector : public std::vector<T>
  {
  public:
    host_vector() : std::vector<T>()
    { }

    host_vector(const host_vector<T> &vec) :
      std::vector<T> (vec)
    {  }

    host_vector(int size) : 
      std::vector<T> (size)
    {  }

    host_vector(const device_vector<T> &vec) :
      std::vector<T> (vec.size())
    {
#ifdef QMC_CUDA
      cudaMemcpy (&((*this)[0]), &(vec[0]), this->size()*sizeof(T), 
		  cudaMemcpyDeviceToHost);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
	fprintf (stderr, "CUDA error in host_vector::copy constructor():\n  %s\n",
		 cudaGetErrorString(err));
	abort();
      }
#endif
    }


    host_vector& 
    operator=(const host_vector<T> &vec)
    {
      if (this->size() != vec.size())
	resize(vec.size());
#ifdef QMC_CUDA
      cudaMemcpy (&((*this)[0]), &(vec[0]), this->size()*sizeof(T), 
		  cudaMemcpyHostToDevice);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
	fprintf (stderr, "CUDA error in host_vector::operator=():\n  %s\n",
		 cudaGetErrorString(err));
	abort();
      }

#endif
      return *this;
    }

    host_vector& 
    operator=(const device_vector<T> &vec)
    {
      if (this->size() != vec.size())
	resize(vec.size());
#ifdef QMC_CUDA
      cudaMemcpy (&((*this)[0]), &(vec[0]), this->size()*sizeof(T), 
		  cudaMemcpyDeviceToHost);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
	fprintf (stderr, "CUDA error in host_vector::operator=():\n  %s\n",
		 cudaGetErrorString(err));
	abort();
      }
#endif
      return *this;
    }
  };

  template<typename T>
  device_vector<T>::device_vector(const host_vector<T> &vec) :
    data_pointer(NULL), current_size(0), alloc_size(0)
  {
#ifdef QMC_CUDA
    cudaMemcpy (&((*this)[0]), &(vec[0]), this->size()*sizeof(T), 
		cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf (stderr, "CUDA error in host_vector::operator=():\n  %s\n",
	       cudaGetErrorString(err));
      abort();
    }

#endif
  
  }



      
}

#endif
