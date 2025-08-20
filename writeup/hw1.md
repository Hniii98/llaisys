### 1. void load(const void *src);

load 函数的语义是将src所指向的内存地址数据拷贝到当前tensor拥有的内存空间,也就是_memory所指向的内存地址空间。因为当前runtime所处的环境可能是CPU也可能是Device, 所以需要根据device_type确定调用的内存拷贝api.

内存拷贝函数的语义逐字节拷贝,并不关心数据在运行时环境的数据类型,因此需要通过Tensor中维护的类型和数据个数指示api拷贝正确的数据大小.


### 2. bool isContiguous() const; 

判断数据是否连续就是要判断数学表达式： 
- 最低维的stride = 1.
  - 因为连续存储所以最低维度的数据间隙应该为1
- 是否满足递推式 stride[k+1] = stride[k] * shape[k]



### 3. tensor_t view(const std::vector<size_t> &shape) const;

view函数的语义是在不改变数据存储结构的前提下,改变数据的访问方式。

这个函数的关键点其实是在判断view操作是否合法,因此需要检查
- tensor是否contiguous
  - 如果当前tensor不连续了, 就无法在不改变存储结构的前提下按照 stride[k+1] = stride[k] * shape[k] 的逻辑去计算出对应stride的函数来支持访问。
- 大小是否一致

如果上述的检查通过,那么就只需要创建一个新的元信息然后复用当前Tensor的存储空间就可以了。


### 4. tensor_t permute(const std::vector<size_t> &order) const;

首先检查给定order的维度是否一致,然后通过order给出的索引对应更改stride和shape创建新的元信息


### 5. tensor_t slice(size_t dim, size_t start, size_t end) const;

根据slice对应dim的stride给tensor设置正确的offset即可