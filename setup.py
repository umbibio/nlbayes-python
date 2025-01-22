from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "nlbayes.ModelORNOR",
        sources=[
            "nlbayes/ModelORNOR.pyx",
            "core/src/Beta.cpp",
            "core/src/Dirichlet.cpp",
            "core/src/GraphBase.cpp",
            "core/src/GraphORNOR.cpp",
            "core/src/HNode.cpp",
            "core/src/HNodeORNOR.cpp",
            "core/src/HParentNode.cpp",
            "core/src/ModelBase.cpp",
            "core/src/ModelORNOR.cpp",
            "core/src/Multinomial.cpp",
            "core/src/NodeDictionary.cpp",
            "core/src/RVNode.cpp",
            "core/src/SNode.cpp",
            "core/src/TNode.cpp",
            "core/src/XNode.cpp",
            "core/src/YDataNode.cpp",
            "core/src/YNoiseNode.cpp"
        ],
        include_dirs=["core/include"],
        libraries=["gsl", "gslcblas", "m"],
        language="c++",
        extra_compile_args=["-std=c++17"],  # Updated to C++17
        depends=["core/include/*.h"],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)
