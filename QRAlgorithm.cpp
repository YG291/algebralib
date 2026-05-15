#include <iostream>
#include <cstdlib>

//continuous matrix formatted as [column 1, column 2, ..., column n-1]

int sign(double number)
{
    if (number > 0) {
        return 1;
    }
    if (number < 0) {
        return -1;
    }
    if (std::abs(number) < 1e-18) return 1;
    return 1;
}

double* leftMultiply(double* matrix_1, double* matrix_2, int m, int n, int d)
{
    // matrix_1 is mxn, matrix_2 is nxd -> product is m*d matrix
    // row element is [column * height + row]
    double* product{ new double[m * d] {} };

    for (int j = 0; j < d; j++)
    {
        for (int i = 0; i < m; i++)
        {
            double sum = 0;
            for (int k = 0; k < n; k++)
            {
                sum += matrix_1[k * m + i] * matrix_2[j * n + k];
            }
            product[j * m + i] = sum;
        }
    }
    return product;
}

double* transpose(double* matrix, int m, int n)
{
    int index = 0;
    double* transposed{ new double[n * m] {} };
    for (int i = 0; i < m; i++) //number of rows in matrix
    {
        for (int j = i; j < m * n; j += m)
        {
            transposed[index] = matrix[j];
            index++;
        }
    }
    return transposed;
}

double* matsum(double* matrix_1, double* matrix_2, int m, int n)
//let's assume both matricies are mxn
{
    double* sumMatrix{ new double[m * n] };
    for (int i = 0; i < m * n; i++)
    {
        sumMatrix[i] = matrix_1[i] + matrix_2[i];
    }
    return sumMatrix;
}

double* scalarmult(double* matrix, int m, int n, double c)
//different from scalar prod
{
    for (int i = 0; i < m * n; i++)
    {
        matrix[i] = c * matrix[i];
    }
    return matrix;
}

double* make_id(double* matrix, int n)
//let's assume matrix is nxn, already prefilled zeros -> could be O(n)
{
    for (int i = 0; i < n; i++)//column number
    {
        //i*n+i is max n^2-1
        matrix[i * n + i] = 1;
    }
    return matrix;
}

double* completeReflector(double* matrix, double* reflector, int n, int m)
{
    //matrix is the final reflector, matrix is nxn, reflector is mxm
    //let's assume that matrix is prefilled with zeros
    for (int i = 0; i < n - m; i++)
    {
        matrix[i * n + i] = 1;
    }
    int reflectorIndex = 0;
    for (int j = 0; j < m; j++)
    {
        for (int k = (n - m + 1 + j) * n - m; k < (n - m + 1 + j) * n; k++)
        {
            matrix[k] = reflector[reflectorIndex];
            reflectorIndex += 1;
        }
    }
    return matrix;
}

double* hessenbergReduce(double* matrix, int n)
{
    //absolutely terrible big O asymtotic runtime. will fix later
    for (int i = 0; i < n - 2; i++)
    {
        double* x{ new double[n - i - 1] {} };
        double* a{ new double[n - i - 1] {} };
        double norm{};
        for (int j = i * n + i + 1; j < i * n + n; j++)
        {
            x[j - i * n - i - 1] = matrix[j];
            a[j - i * n - i - 1] = x[j - i * n - i - 1];
            norm = norm + matrix[j] * matrix[j];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-18) {
            delete[] x;
            delete[] a;
            continue;
        }
        a[0] = x[0] + sign(x[0]) * norm;
        double* identity{ new double[(n - i - 1) * (n - i - 1)] {} };
        identity = make_id(identity, n - i - 1);

        double* atranspose = transpose(a, n - i - 1, 1);
        double* denom = leftMultiply(atranspose, a, 1, n - i - 1, 1);
        double div{ 1 / *denom };
        delete[] denom;

        double* proja = scalarmult(leftMultiply(a, atranspose, n - i - 1, 1, n - i - 1), n - i - 1, n - i - 1, div);
        proja = scalarmult(proja, n - i - 1, n - i - 1, -2);
        double* subreflector{ new double[(n - i - 1) * (n - i - 1)] {} };
        double* fullid{ new double[(n - i - 1) * (n - i - 1)] {} };
        fullid = make_id(fullid, n - i - 1);

        double* mm = subreflector;
        subreflector = matsum(fullid, proja, n - i - 1, n - i - 1);
        delete[] mm;

        double* housereflector{ new double[n * n] {} };
        housereflector = completeReflector(housereflector, subreflector, n, n - i - 1);

        mm = matrix;
        matrix = leftMultiply(housereflector, matrix, n, n, n);
        delete[] mm;
        mm = matrix;
        matrix = leftMultiply(matrix, housereflector, n, n, n);
        delete[] mm;

        delete[] x;
        delete[] a;
        delete[] proja;
        delete[] atranspose;
        delete[] subreflector;
        delete[] housereflector;
        delete[] identity;
        delete[] fullid;
    }
    return matrix;
    //doesn't modify the same matrix which is kinda problematic
    //memory inefficiencies are because we do not in-place matrix multiply
}

void givensRotate(double* matrix, int n)
{
    //idea: G is the 2x2 rotation matrix, 'complete it' as an nxn, 
    //with G as top left, rest is I_n (mathematically)
    //We do (GnGn-1...G2G1A)(G
    double* cos{ new double[n - 1] };
    double* sin{ new double[n - 1] };
    for (int i = 0; i < n - 1; i++)//# of rows
    {
        double diagonal{ matrix[i * n + i] };
        double subdiagonal{ matrix[i * n + i + 1] };
        double r = std::sqrt(diagonal * diagonal + subdiagonal * subdiagonal);
        if (r < 1e-18) {
            cos[i] = 1.0;
            sin[i] = 0.0;
            continue;
        }
        double x1 = diagonal / r;
        double x2 = subdiagonal / r;
        double x3 = -x2;
        cos[i] = x1;
        sin[i] = x2;
        for (int j = 0; j < n; j++) // G*M, multiplying rows of M
        {
            double el1 = matrix[j * n + i];
            double el2 = matrix[j * n + i + 1];
            matrix[j * n + i] = x1 * el1 + x2 * el2;
            matrix[j * n + i + 1] = x3 * el1 + x1 * el2;
        }
    }

    for (int i = 0; i < n-1; i++)//# of rows
    {
        double x1 = cos[i];
        double x2 = sin[i];
        double x3 = -x2;
        for (int k = 0; k < n; k++) // M*G^T, multiplying cols of M
        {
            //in column-position wise pairs (i.e. third pos in column)
            double el1 = matrix[i * n + k];
            double el2 = matrix[(i + 1) * n + k];
            matrix[i * n + k] = x1 * el1 + x2 * el2;
            matrix[(i + 1) * n + k] = x3 * el1 + x1 * el2;

        }
    }
    delete[] cos;
    delete[] sin;
}

double* QRIterate(double* matrix, int n)
{
    matrix = hessenbergReduce(matrix, n);
    double* mm{};
    for (int i = 0; i < 100*n; i++)
    {
        givensRotate(matrix, n);
    }
    return matrix;
}

//issue with the current implementation is that it does not account for complex eigenvalues