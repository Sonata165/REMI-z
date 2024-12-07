python setup.py sdist

twine upload dist/*

rm -rf dist build REMI_z.egg-info
