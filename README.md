<!-- TODO: CHANGE ALL INSTANCES OF "TEMPLATE-README" IN ENTIRE PROJECT TO YOUR PROJECT TITLE-->
# Hype AI


<div align="center">

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/CogitoNTNU/TEMPLATE-README/ci.yml)
![GitHub top language](https://img.shields.io/github/languages/top/CogitoNTNU/TEMPLATE-README)
![GitHub language count](https://img.shields.io/github/languages/count/CogitoNTNU/TEMPLATE-README)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Version](https://img.shields.io/badge/version-0.0.1-blue)](https://img.shields.io/badge/version-0.0.1-blue)

<img src="docs/images/project-logo.webp" width="50%" alt="Cogito Project Logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>


<details> 
<summary><b>📋 Table of contents </b></summary>

- [Hype AI](#hype-ai)
  - [Description](#description)
  - [Getting started](#getting-started)
    - [Run using docker (prod):](#run-using-docker-prod)
    - [Installation (development):](#installation-development)
    - [Run (development)](#run-development)
    - [Configuration](#configuration)
    - [Prerequisites](#prerequisites)
  - [Usage](#usage)
  - [Testing](#testing)
  - [Team](#team)
    - [License](#license)

</details>

## Description 
HypeAI automates the creation and uploading of videos to social media like TikTok. The videos are generated using AI.

Supported generator are:

* Quiz Videos 🧠
* Guess The Celebrity 🌟

Supported uploaders are:

* TikTok



## Getting started

### Run using docker (prod):
```Bash
make run
```

### Installation (development):
```Bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd tiktok-uploader
hatch build
pip install -e .
cd ..
```

### Run (development)
```Bash
python main.py
```
### Configuration

<!-- TODO: Describe how to configure the project (environment variables, config files, etc.).

### Configuration
Create a `.env` file in the root directory of the project and add the following environment variables:

```bash
OPENAI_API_KEY = 'your_openai_api_key'
MONGODB_URI = 'your_secret_key'
```
-->

### Prerequisites
- Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- Make
- Docker


## Usage
To run the project, run the following command from the root directory of the project:
```bash

```
<!-- TODO: Instructions on how to run the project and use its features. -->

## Testing
To run the test suite, run the following command from the root directory of the project:
```bash

```

## Team
This project would not have been possible without the hard work and dedication of all of the contributors. Thank you for the time and effort you have put into making this project a reality.


<table align="center">
    <tr>
        <!--
        <td align="center">
            <a href="https://github.com/NAME_OF_MEMBER">
              <img src="https://github.com/NAME_OF_MEMBER.png?size=100" width="100px;" alt="NAME OF MEMBER"/><br />
              <sub><b>NAME OF MEMBER</b></sub>
            </a>
        </td>
        -->
    </**tr**>
</table>

![Group picture](docs/img/team.png)


### License
------
Distributed under the MIT License. See `LICENSE` for more information.
