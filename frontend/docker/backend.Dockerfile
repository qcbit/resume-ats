FROM node:23-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the backend source code
COPY . .

# Expose the backend port (adjust if needed)
EXPOSE 5000

# Start the backend service
CMD ["npm", "start"]