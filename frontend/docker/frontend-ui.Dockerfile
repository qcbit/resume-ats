FROM node:23-alpine AS build

WORKDIR /app

# Accept API URL as a build argument, default to http://resume-backend:5000
ARG VITE_REACT_APP_API_URL=http://resume-backend:5000
ENV VITE_REACT_APP_API_URL=${VITE_REACT_APP_API_URL}

COPY . .
RUN npm install
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]