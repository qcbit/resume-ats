import User from '../models/user.model.js';
import jwt from 'jsonwebtoken';
import { expressjwt as expressJwt } from 'express-jwt';
import config from '../../config/config.js';

// This function handles user sign-in by verifying the user's credentials
// and returns a generated JWT token if the credentials are valid.
const signin = async (req, res) => {
  try {
    let user = await User.findOne({ 'email': req.body.email });
    if (!user) {
      return res.status('401').json({ error: "User not found" });
    }

    if (!user.authenticate(req.body.password)) {
      return res.status('401').send({ error: "Email and password don't match" });
    }

    const token = jwt.sign({ _id: user._id }, config.jwtSecret);
    res.cookie('t', token, { expire: new Date() + 9999 });

    return res.json({
      token,
      user: {
        _id: user._id,
        name: user.name,
        email: user.email,
      }
    })
  } catch (err) {
    return res.status('401').json({ error: "Could not sign in" });
  }
};

// This function handles user sign-out by clearing the JWT token cookie.
const signout = (req, res) => {
  res.clearCookie("t");
  return res.status('200').json({ message: "Signed out successfully" });
};

// Middleware to check if the user is signed in
// throws an error if the JWT token is not valid or expired.
const requireSignin = expressJwt({
  secret: config.jwtSecret,
  userProperty: 'auth',
  algorithms: ['HS256']
});

// Middleware to check if the user has authorization to perform certain actions
const hasAuthorization = (req, res, next) => {
  const authorized = req.profile && req.auth && req.profile._id == req.auth._id;
  if (!authorized) {
    return res.status('403').json({ error: "User is not authorized" });
  }
  next();
};

export default {
  signin,
  signout,
  requireSignin,
  hasAuthorization
};