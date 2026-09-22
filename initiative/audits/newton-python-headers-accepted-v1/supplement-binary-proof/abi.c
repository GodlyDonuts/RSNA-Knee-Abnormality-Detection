#include <Python.h>
#include <stdio.h>
int main(void){printf("%s|%u|%zu|%zu\n",PY_VERSION,PY_VERSION_HEX,sizeof(void*),sizeof(PyObject));return 0;}
