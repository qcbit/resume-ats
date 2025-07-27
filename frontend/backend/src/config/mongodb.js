import mongoose from 'mongoose';
import dotenv from 'dotenv';
import path from 'path';

// Load .env file from the project root
dotenv.config({ path: path.resolve(process.cwd(), '../../.env') });

export const connectDB = async () => {
  try {
    const mongoURI = process.env.DATABASE_URL || `mongodb://${process.env.MONGO_USER}:${process.env.MONGO_PASSWORD}@localhost:27017/${process.env.MONGO_INITDB_DATABASE}?authSource=${process.env.MONGO_INITDB_DATABASE}`;
    await mongoose.connect(mongoURI);
    console.log("MongoDB connected successfully");
  } catch (err) {
    console.error("Error connecting to MongoDB:", err);
    process.exit(1);
  }
};
