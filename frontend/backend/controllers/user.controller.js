import User from '../models/user.model.js';
import extend from 'lodash/extend.js';
import errorHandler from '../helpers/dbErrorHandler.js';

// Create a new user
const create = async (req, res) => {
  const user = new User(req.body);
  try {
    await user.save();
    return res.status(200).json({
      message: "Successfully signed up!"
    })
  } catch (err) {
    return res.status(400).json({
      error: errorHandler.getErrorMessage(err)
    })
  }
};

// List all users
const list = async (req, res) => {
  try {
    let users = await User.find().select('name email updated created');
    res.json(users);
  } catch (err) {
    return res.status(400).json({
      error: errorHandler.getErrorMessage(err)
    });
  }
};

// Retrieve user by ID
const userByID = async (req, res, next, id) => {
  try {
    let user = await User.findById(id);
    if (!user) {
      return res.status('400').json({
        error: "User not found"
      });
    }
    req.profile = user;
    next();
  } catch (err) {
    return res.status('400').json({
      error: "Could not retrieve user"
    });
  }
};

// Read user profile
const read = (req, res) => {
  req.profile.hashed_password = undefined; // Remove sensitive data
  req.profile.salt = undefined;            // Remove sensitive data
  return res.json(req.profile);
};

// Update user information
const update = async (req, res) => {
  try {
    let user = req.profile;
    user = extend(user, req.body); // Extend user with new data
    user.updated_at = Date.now();
    await user.save();
    user.hashed_password = undefined; // Remove sensitive data
    user.salt = undefined;            // Remove sensitive data
    res.json(user);
  } catch (err) {
    return res.status(400).json({
      error: errorHandler.getErrorMessage(err)
    });
  }
};

// Delete user
const remove = async (req, res) => {
  try {
    let user = req.profile;
    let deletedUser = await user.remove();
    deletedUser.hashed_password = undefined; // Remove sensitive data
    deletedUser.salt = undefined;            // Remove sensitive data
    res.json(deletedUser);
  } catch (err) {
    return res.status(400).json({
      error: errorHandler.getErrorMessage(err)
    });
  }
};

export default { create, list, userByID, read, update, remove };