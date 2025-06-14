const config = {
  env: process.env.NODE_ENV || 'development',
  port: process.env.PORT || 3000,
  jwtSecret: process.env.JWT_SECRET || 'default',
  mongoUri: process.env.MONGODB_URI ||
    process.env.MONGO_HOST ||
    'mongodb://app-user:app123@' + (process.env.IP || 'localhost') + ':' +
    (process.env.MONGO_PORT || '27017') + '/app?directConnection=true&replicaSet=mongodb-replica-set&authSource=app',
};

export default config;