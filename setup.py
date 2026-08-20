from setuptools import setup, find_packages
import os


version = '0.7.3'

readme_path = 'Readme.md'
long_description = open(readme_path, encoding='utf-8').read() if os.path.exists(readme_path) else ""


setup(
    name='REMI-z',  # 项目名称
    version=version,  # 版本号
    author='Longshen Ou',  # 作者姓名
    author_email='oulongshen@gmail.com',  # 邮箱地址
    description='Manipulate your MIDI file in bar level, and converting between MIDI and REMI-z format.',
    long_description=open('Readme.md').read(),  # 从 Readme.md 加载详细描述
    long_description_content_type='text/markdown',  # README 格式
    url='https://github.com/Sonata165/REMI-z',  # 项目主页 URL
    # Declares the flat layout explicitly: the package root IS the project root.
    # Semantically a no-op (that is already the default), but it is what makes
    # `pip install -e .` produce an IDE-resolvable editable install. setuptools>=64
    # picks its editable strategy in `_select_strategy`:
    #     if set(package_dir) == {""} and has_simple_layout or is_compat_mode:
    #         return _StaticPth(...)   # a .pth holding a plain directory path
    #     return _TopLevelFinder(...)  # a .pth holding `import ..._finder; install()`
    # Without this line package_dir == {} != {""}, so we got _TopLevelFinder — an
    # import HOOK registered at interpreter start. That works at runtime but is
    # invisible to static analysers (Pylance/Pyright never execute .pth files, they
    # only honour ones containing a path), which is why VSCode reported every
    # `remi_z` import as unresolved while the code ran fine.
    package_dir={'': '.'},
    packages=find_packages(),  # 自动查找所有包含 `__init__.py` 的包
    install_requires=[  # 项目的依赖项
        'miditoolkit>=1.0.1',
        'music21>=8.3.0',
        'pretty_midi>=0.2.10',
        'pyyaml>=6.0.2',
    ],
    classifiers=[  # 分类器，描述项目的适用性
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Sound/Audio :: MIDI',
    ],
    python_requires='>=3.7',  # 支持的最低 Python 版本
    license='MIT',  # 项目许可证
)