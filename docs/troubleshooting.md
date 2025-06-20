### Initial Mongo database and authorized user were not created
The mongo-init.js script isn't being executed, and the most common reason for this is that MongoDB has already been initialized in a previous run, and the data is persisting in a Docker volume. The initialization scripts in /docker-entrypoint-initdb.d/ only run when the database is started for the **very first time** with an empty data directory (/data/db inside the container).

Here's a step-by-step guide to troubleshoot and fix this:

1. Ensure the Init Script Runs (Clear Persistent Data):

You need to remove the existing MongoDB data volume so that MongoDB re-initializes and runs your script.

Stop the MongoDB container:

```bash
docker compose -f <docker-compose.yaml> down
```

Identify and remove the Docker volume: The docker-compose.yaml might define a volume like this:
```yaml
volumes:
  mongodata:
    driver: local
# ...
services:
  mongo-db:
    # ...
    volumes:
      # ...
      - mongodata:/data/db
```

Docker Compose typically names volumes projectname_volumename, so the volume name would be resume-ats_mongodata. You can list all volumes to find the exact name:

```bash
docker volume ls
```

Remove it:

```bash
docker volume rm resume-ats_mongodata
``

2. Restart MongoDB:

```bash
docker compose -f <docker-compose.yaml> up -d
```

3. Check Logs for Initialization Script Output:

```bash
docker logs mongo-db
You should see the print() messages from your mongo-init.js script, like: mongo-init.js: Starting initialization... mongo-init.js: User 'mongouser' created. ...and so on, without errors related to the script.
```

4. Test Connections:

Connect as mongouser to resumeatsusers:

```bash
mongosh "mongodb://mongouser:mongopassword@localhost:27017/resumeatsusers"
```

Inside mongosh:

```javascript
show tables;
db.users.find();
```

You should see the users collection and the sample document.

Connect as admin (root user) to the admin database:

```bash
mongosh "mongodb://admin:admin123@localhost:27017/admin"
```

Inside mongosh:

```javascript
show dbs;
use resumeatsusers;
show users; // Should show mongouser
show tables; // Should show users collection
```

By ensuring the data volume is cleared, MongoDB will perform a fresh initialization, executing your mongo-init.js script.
