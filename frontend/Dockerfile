# Stage 1: Build the Nuxt.js application
FROM node:20 as builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run generate

# Stage 2: Serve the static files with Nginx
FROM nginx:alpine
# Copy the generated static files from the builder stage
# For Nuxt 3, the output is typically in /app/.output/public
COPY --from=builder /app/.output/public /usr/share/nginx/html

# Expose port 80 (Nginx's default)
EXPOSE 80

# Command to start Nginx, which will serve the static files
CMD ["/usr/sbin/nginx", "-g", "daemon off;"]