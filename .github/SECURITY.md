# Security Policy

## Supported Versions

We actively support the following versions of ADNet with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in ADNet, please follow these steps:

### 1. Do Not Open a Public Issue

Please **do not** report security vulnerabilities through public GitHub issues. This could expose the vulnerability to malicious actors.

### 2. Contact Us Privately

Instead, please report security vulnerabilities by:

- **Email**: Send details to `hello@surajit.de` with the subject line "ADNet Security Vulnerability"
- **GitHub Security Advisory**: Use GitHub's private vulnerability reporting feature

### 3. Information to Include

When reporting a vulnerability, please include:

- **Description**: A clear description of the vulnerability
- **Impact**: What could an attacker achieve by exploiting this vulnerability?
- **Reproduction**: Steps to reproduce the vulnerability
- **Affected versions**: Which versions of ADNet are affected
- **Suggested fix**: If you have ideas for how to fix the issue

### 4. Response Timeline

We will respond to security vulnerability reports as follows:

- **Initial response**: Within 48 hours of receiving the report
- **Assessment**: We will assess the vulnerability within 5 business days
- **Fix timeline**: We aim to release a fix within 30 days for critical vulnerabilities
- **Public disclosure**: We will coordinate public disclosure timing with you

### 5. Recognition

We believe in responsible disclosure and will:

- Acknowledge your contribution in our security advisory (if you wish)
- Credit you in our release notes (if you wish)
- Work with you on appropriate disclosure timing

## Security Best Practices

When using ADNet in production environments, please follow these security best practices:

### Data Security
- **Dataset protection**: Ensure proper access controls for dataset files
- **Model security**: Protect trained models from unauthorized access
- **Input validation**: Validate all input data before processing

### Environment Security
- **Dependencies**: Keep all dependencies up to date
- **Python environment**: Use virtual environments and pin dependency versions
- **System security**: Follow your organization's security policies for the deployment environment

### Network Security
- **API endpoints**: Secure any API endpoints that use ADNet
- **Data transmission**: Use encrypted connections for data transfer
- **Access control**: Implement proper authentication and authorization

## Security Features

ADNet includes several security features:

### Input Validation
- All dataset loaders include input validation
- Image dimensions and formats are checked
- Annotation data is validated before processing

### Safe Defaults
- Default configurations prioritize security
- Optional features that could pose security risks are disabled by default

### Dependency Management
- We regularly audit our dependencies for known vulnerabilities
- Automated dependency updates through Dependabot
- Minimal dependency footprint

## Known Security Considerations

### Dataset Handling
- **Large files**: Be aware of potential DoS through extremely large files
- **Malformed data**: Dataset loaders include validation, but always verify data sources
- **Path traversal**: Be cautious with user-provided file paths

### Model Inference
- **Resource consumption**: Models can consume significant GPU/CPU resources
- **Memory usage**: Large models may require substantial memory
- **Input size limits**: Very large inputs could cause resource exhaustion

## Security Updates

Security updates will be:

- Released as patch versions (e.g., 0.1.1, 0.1.2)
- Documented in our changelog with severity levels
- Announced through GitHub releases and security advisories
- Backported to supported versions when possible

## Contact

For security-related questions or concerns:

- **Security issues**: `hello@surajit.de`
- **General questions**: Open a GitHub issue
- **Project maintainer**: [@surajroland](https://github.com/surajroland)

## Acknowledgments

We would like to thank the security researchers and community members who have responsibly disclosed vulnerabilities to help make ADNet more secure.

---

Last updated: August 2024