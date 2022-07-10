from setuptools import setup, find_packages


setup(
    name =          'fatbot',
    version =       '0.0.1', 
    url =           "https://github.com/Nelson-iitp/fatbot",
    author =        "Nelson.S",
    author_email =  "nelson_2121cs07@iitp.ac.in",
    description =   ' Fat-Bots in 2D ',
    package_dir =   { '' : 'src'},
    packages =      [package for package in find_packages(where='./src')],
    #classifiers =   []
    install_requires = ["numpy"],
    #include_package_data=True
)

