FROM node:22 as builder

WORKDIR /app

COPY . .

RUN npm ci && npm run generate


FROM nginx:alpine

COPY --from=builder /app/.output/public /usr/share/nginx/html

EXPOSE 80

CMD ["/usr/sbin/nginx", "-g", "daemon off;"]
