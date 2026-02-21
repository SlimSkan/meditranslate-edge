# MediTranslate Edge - Deployment Guide

## Edge Deployment for Offline Clinical Settings

This guide explains how to deploy MediTranslate Edge in resource-constrained environments without internet access.

## Prerequisites

- Docker and Docker Compose installed
- 16GB+ RAM recommended
- 20GB free disk space
- HuggingFace account and token

## Quick Start (Online Setup)

1. **Clone the repository:**
```bash
   git clone https://github.com/YOUR_USERNAME/meditranslate-edge.git
   cd meditranslate-edge
```

2. **Set your HuggingFace token:**
```bash
   export HF_TOKEN=hf_your_token_here
```

3. **Build and run:**
```bash
   docker-compose up --build
```

4. **Access the application:**
   - Open browser to: http://localhost:8501

## Offline Deployment

For truly offline deployment in clinical settings:

### Phase 1: Online Preparation (One-time)

1. **Build the container with internet access:**
```bash
   docker-compose up --build
```
   
   This downloads MedGemma (~8GB) and caches it in the Docker volume.

2. **Save the container:**
```bash
   docker save meditranslate-edge:latest > meditranslate-edge.tar
```

3. **Save the model cache volume:**
```bash
   docker run --rm -v meditranslate_model_cache:/cache -v $(pwd):/backup \
     alpine tar czf /backup/model_cache.tar.gz -C /cache .
```

### Phase 2: Offline Transfer

Transfer these files to the offline environment:
- `meditranslate-edge.tar` (~10GB)
- `model_cache.tar.gz` (~8GB)
- `docker-compose.yml`

### Phase 3: Offline Deployment

1. **Load the container:**
```bash
   docker load < meditranslate-edge.tar
```

2. **Restore the model cache:**
```bash
   docker volume create meditranslate_model_cache
   docker run --rm -v meditranslate_model_cache:/cache -v $(pwd):/backup \
     alpine tar xzf /backup/model_cache.tar.gz -C /cache
```

3. **Run offline:**
```bash
   docker-compose up
```

## Hardware Requirements

### Minimum:
- CPU: 4 cores
- RAM: 16GB
- Storage: 20GB
- OS: Linux, Windows 10+, macOS

### Recommended:
- CPU: 8+ cores
- RAM: 32GB
- GPU: 8GB+ VRAM (optional, improves speed)
- Storage: 50GB SSD

## Performance

- **First run:** 5-10 minutes (model loading)
- **Subsequent runs:** ~30 seconds startup
- **Translation speed:** 2-5 seconds per request (CPU)
- **Translation speed:** <1 second per request (GPU)

## Security Considerations

- Container runs as non-root user
- No data persistence by default (HIPAA consideration)
- Model runs entirely locally
- No network calls after initial setup
- Logs can be disabled for privacy

## Troubleshooting

**Container won't start:**
- Check Docker is running: `docker ps`
- Check logs: `docker-compose logs`

**Out of memory:**
- Increase Docker memory limit (Docker Desktop settings)
- Close other applications
- Use model quantization (future enhancement)

**Slow performance:**
- Use GPU if available
- Reduce concurrent users
- Increase system RAM

## Cost Analysis

**Cloud API alternative:**
- ~$0.01 per translation
- 1000 translations/month = $120/year
- 5 years = $600

**Edge deployment:**
- Hardware: $500-1000 (one-time)
- Electricity: ~$50/year
- 5 years = $550-1250 total

**Break-even:** ~6-12 months of operation

## Support

- GitHub Issues: [repository URL]
- Demo: https://www.kaggle.com/code/slimkhaled/medgemmainterface
- Email: [your email if you want]

## License

MIT License - See LICENSE file
