# What does *unrolling loops* even mean??
It's an optimization technique that tries to reduce the overhead of loop control by replicating the loop body multiple times. This is done by replacing the loop with a sequence of statements that are equivalent to the loop body, but are executed sequentially. This can be done by the compiler or by the programmer.

## Examples
Suppose we want to add the elements of one array together. We could write a loop like this:
```c
int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

int sum(int *array) {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += arr[i];
    }

    return sum;
}
```
This loop does ten iterations! This can be reduced.

The loop could be *partially unrolled* like this:
```c
int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

int sum(int *array) {
    int sum = 0;
    for (int i = 0; i < 10; i += 2) {
        sum += arr[i];
        sum += arr[i + 1];
    }

    return sum;
}
```

What this effectively does is reduce the number of iterations by half. This is because the loop body is executed twice per iteration.

A loop can also be *fully unrolled* like this:
```c
int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

int sum(int *array) {
    int sum = 0;

    sum += arr[0];
    sum += arr[1];
    sum += arr[2];
    sum += arr[3];
    sum += arr[4];
    sum += arr[5];
    sum += arr[6];
    sum += arr[7];
    sum += arr[8];
    sum += arr[9];

    return sum;
}
```

This loop is unrolled completely, meaning that there are no iterations at all. This can be beneficial in some cases, but it can also increase the size of the code and reduce the effectiveness of the cache.

## Advantages
- **Reduced Loop Overhead**: Fewer iterations mean fewer evaluations of the loop condition and fewer increments/decrements of the loop counter.
- **Increased Performance**: More operations are performed per loop iteration, which can lead to better utilization of the CPU cache and pipeline.

## Disadvantages
- **Increased Code Size**: Unrolling a loop can significantly increase the size of the compiled code, especially if the loop body is large or the loop is unrolled many times.
- **Complexity in Maintenance**: The unrolled code is often harder to read and maintain.

## Using a MLIR dialect for loop unrolling
The MLIR dialect provides a way to represent loops in a structured form that can be analyzed and transformed by various optimization passes. The `affine.for` operation in the Affine dialect represents a loop with a known trip count and a known step size. This makes it possible to unroll loops in a principled way using MLIR.

Here is an example of how a loop can be unrolled using MLIR:
```mlir
func.func @sum_buffer(%buffer: memref<4xi32>) -> (i32) {
  %sum_0 = arith.constant 0 : i32
  %sum = affine.for %i = 0 to 4 iter_args(%sum_iter = %sum_0) -> i32 {
    %t = affine.load %buffer[%i] : memref<4xi32>
    %sum_next = arith.addi %sum_iter, %t : i32
    affine.yield %sum_next : i32
  }
  return %sum : i32
}
```
