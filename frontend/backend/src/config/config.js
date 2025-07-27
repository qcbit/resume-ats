const config = {
  env: process.env.NODE_ENV || 'development',
  port: process.env.PORT || 3000,
  jwtSecret: process.env.JWT_SECRET || 'default',
  mongoUri: process.env.MONGO_HOST ||
    'mongodb://' + (process.env.IP) || 'localhost' + ':' +
    (process.env.MONGO_PORT) || '27017/' + (process.env.MONGO_DB || 'myapp')
};

export default config;