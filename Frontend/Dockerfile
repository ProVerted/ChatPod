# Multi-stage production Dockerfile for ChatPod React Frontend
FROM node:22-alpine AS builder

WORKDIR /app

# Install dependencies
COPY frontend/package*.json ./
RUN npm install --legacy-peer-deps

# Build static assets
COPY frontend/ ./
ARG REACT_APP_API_BASE_URL=http://localhost:8000
ENV REACT_APP_API_BASE_URL=$REACT_APP_API_BASE_URL
RUN npm run build

# Production web server (Nginx)
FROM nginx:alpine AS runner

COPY --from=builder /app/build /usr/share/nginx/html

# Custom Nginx configuration for single page application routing
RUN echo 'server { \
    listen 80; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
