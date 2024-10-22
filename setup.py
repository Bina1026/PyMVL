from setuptools import setup, find_packages

setup(
    name='pyMVL',  # 패키지 이름
    version='0.1.0',  # 패키지 버전
    packages=find_packages(where="."),  # 패키지를 찾는 경로
    package_dir={'': 'pyMVL'},  # 패키지 디렉토리 설정
    install_requires=[
        'numpy>=1.18.0',
        'pandas>=1.0.0'
    ],  # 필요한 패키지를 여기에 추가
    author='Hyunbin Joo',
    author_email='jooben12345@gmail.com',
    description='package for Radiomics Visualization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/pyMVL',  # 깃허브 URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)