// mongo-init.js
// This script initializes the MongoDB database with a sample user collection.
// It is executed when the MongoDB container starts.
// This script is run automatically by the MongoDB Docker image when the container starts.
// The database name is specified in the environment variable MONGO_INITDB_DATABASE.
// The script will create a 'users' collection and insert a sample user document.
// The script uses the MongoDB shell commands to perform the initialization.
// The script is executed in the context of the MongoDB container.
// Note: It is not necessary to create the database explicitly, as MongoDB will create it when the first document is inserted.
// Also, the admin user is created automatically by the MongoDB Docker entrypoint script.
// The official MongoDB Docker image has a specific way to handle initialization scripts.
// When it starts for the first time with an empty data volume, it will check for the MONGO_INITDB_ROOT_USERNAME 
// and MONGO_INITDB_ROOT_PASSWORD environment variables. It uses these to create an admin user.
// The mongo-init.js script runs after the admin user is created, so you can use the admin user to perform operations on the database.
db.getSiblingDB('admin').auth(
    process.env.MONGO_INITDB_ROOT_USERNAME,
    process.env.MONGO_INITDB_ROOT_PASSWORD
);
db.getSiblingDB(process.env.MONGO_INITDB_DATABASE).createUser({
    user: process.env.MONGO_USER,
    pwd: process.env.MONGO_PASSWORD,
    roles: [{ role: 'readWrite', db: process.env.MONGO_INITDB_DATABASE }]
});

print(`mongo-init.js: Initialization for database '${db.getName()}' complete.`);