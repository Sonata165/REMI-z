from setuptools import setup, find_packages

setup(
    name='REMI-z',  # 项目名称
    version='0.5.2',  # 版本号
    author='Longshen Ou',  # 作者姓名
    author_email='oulongshen@gmail.com',  # 邮箱地址
    description='Manipulate your MIDI file in bar level, and converting between MIDI and REMI-z format.',
    long_description=open('README.md').read(),  # 从 README.md 加载详细描述
    long_description_content_type='text/markdown',  # README 格式
    url='https://github.com/Sonata165/REMI-z',  # 项目主页 URL
    packages=find_packages(),  # 自动查找所有包含 `__init__.py` 的包
    install_requires=[  # 项目的依赖项
        'miditoolkit==1.0.1',
        'music21==9.3.0',
        'pretty_midi==0.2.10',
        'pyyaml==6.0.2',
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