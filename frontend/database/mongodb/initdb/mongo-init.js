// The print statements are seen in the logs of the MongoDB container
print(`mongo-init.js: Starting initialization for database '${db.getName()}'.`);

db.createUser({
    user: "mongouser", // For production, use a strong, environment-injected username
    pwd: "mongopassword", // For production, use a strong, environment-injected password
    roles: [{ role: "readWrite", db: "resumeatsusers" }]
});
print("mongo-init.js: User 'mongouser' created.");

db.createCollection("users");
print("mongo-init.js: Collection 'users' created.");

// Database is not created until a document is inserted
db.users.insertOne({
    name: "user1",
    email: "user1.test@email.com",
    password: "password1" // Example data, ensure proper hashing in a real app
});
print("mongo-init.js: Sample document inserted into 'users' collection.");

print(`mongo-init.js: Initialization for database '${db.getName()}' complete.`);





// db = db.getSiblingDB("admin");
// db.auth({
//   user: process.env.MONGO_INITDB_ROOT_USERNAME,
//   password: process.env.MONGO_INITDB_ROOT_PASSWORD
// });

// db.createUser({
//   'user': 'mongouser',
//   'pwd': 'mongopassword',
//   'roles': [
//     "readWriteAnyDatabase",
//   ]
// });

// db = db.getSiblingDB("resumeatsusers");
// db.createCollection('users', {capped: false});
// db.users.insert({name: 'user1', email: 'user1.test@email.com', password: 'password1'});