# from pyspark import SparkContext
# sc = SparkContext.getOrCreate()
# print(sc.version)



import pkg_resources

# Retrieve all installed packages and their versions
installed_packages = [(d.project_name, d.version) for d in pkg_resources.working_set]

# Print each package and version
for package_name, version in installed_packages:
    print(f"{package_name}=={version}")